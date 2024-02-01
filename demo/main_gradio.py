import gradio as gr
import argparse
import time, os, sys
from PIL import Image
import numpy as np
import torch
from typing import List, Literal, Dict, Optional
from draw_utils import draw_points_on_image, draw_mask_on_image
import cv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.streamdiffusion.wrapper import StreamDiffusionWrapper

from models.animatediff.pipelines import I2VPipeline
from omegaconf import OmegaConf

from models.draggan.viz.renderer import Renderer
from models.draggan.gan_inv.lpips.util import PerceptualLoss
import models.draggan.dnnlib as dnnlib
from models.draggan.gan_inv.inversion import PTI

import imageio
import torchvision
from einops import rearrange

# =========================== Model Implementation Start ===================================

def save_videos_grid_255(videos: torch.Tensor, path: str, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        x = x.numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points

def render_view_image(img, drag_markers, show_mask=False):
    img = draw_points_on_image(img, drag_markers['points'])
    if show_mask:
        img = draw_mask_on_image(img, drag_markers['mask'])
    img = np.array(img).astype(np.uint8)
    img = np.concatenate([
        img,
        255 * np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
    ], axis=2)
    return Image.fromarray(img)


def update_state_image(state):
    state['generated_image_show'] = render_view_image(
        state['generated_image'],
        state['drag_markers'][0],
        state['is_show_mask'],
    )
    return state['generated_image_show']


class GeneratePipeline:
    def __init__(
        self, 
        i2i_body_ckpt: str = "checkpoints/diffusion_body/kohaku-v2.1",
        i2i_lora_dict: Optional[Dict[str, float]] = {'checkpoints/i2i/lora/lcm-lora-sdv1-5.safetensors': 1.0},
        prompt: str = "",
        negative_prompt: str = "low quality, bad quality, blurry, low resolution",
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
        use_denoising_batch: bool = True,
        seed: int = 2,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        guidance_scale: float = 1.4,
        delta: float = 0.5,
        do_add_noise: bool = False,
        enable_similar_image_filter: bool = True,
        similar_image_filter_threshold: float = 0.99,
        similar_image_filter_max_skip_frame: float = 10,
    ):
        super(GeneratePipeline, self).__init__()
        self.img2img_model = None
        self.img2video_model = None
        self.img2video_generator = None
        self.sim_ranges = None

        # set parameters
        self.i2i_body_ckpt = i2i_body_ckpt
        self.i2i_lora_dict = i2i_lora_dict
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.frame_buffer_size = frame_buffer_size
        self.width = width
        self.height = height
        self.acceleration = acceleration
        self.use_denoising_batch = use_denoising_batch
        self.seed = seed
        self.cfg_type = cfg_type
        self.guidance_scale = guidance_scale
        self.delta = delta
        self.do_add_noise = do_add_noise
        self.enable_similar_image_filter = enable_similar_image_filter
        self.similar_image_filter_threshold = similar_image_filter_threshold
        self.similar_image_filter_max_skip_frame = similar_image_filter_max_skip_frame

        self.i2v_config = OmegaConf.load('demo/configs/i2v_config.yaml')
        self.i2v_body_ckpt = self.i2v_config.pretrained_model_path
        self.i2v_unet_path = self.i2v_config.generate.model_path
        self.i2v_dreambooth_ckpt = self.i2v_config.generate.db_path

        self.lora_alpha = 0

        assert self.frame_buffer_size == 1
    
    def init_model(self):
        # StreamDiffusion
        self.img2img_model = StreamDiffusionWrapper(
            model_id_or_path=self.i2i_body_ckpt,
            lora_dict=self.i2i_lora_dict,
            t_index_list=[32, 45],
            frame_buffer_size=self.frame_buffer_size,
            width=self.width,
            height=self.height,
            warmup=10,
            acceleration=self.acceleration,
            do_add_noise=self.do_add_noise,
            enable_similar_image_filter=self.enable_similar_image_filter,
            similar_image_filter_threshold=self.similar_image_filter_threshold,
            similar_image_filter_max_skip_frame=self.similar_image_filter_max_skip_frame,
            mode="img2img",
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=self.cfg_type,
            seed=self.seed,
            use_lcm_lora=False,
        )
        self.img2img_model.prepare(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=50,
            guidance_scale=self.guidance_scale,
            delta=self.delta,
        )
        
        # PIA
        self.img2video_model = I2VPipeline.build_pipeline(
            self.i2v_config,
            self.i2v_body_ckpt,
            self.i2v_unet_path,
            self.i2v_dreambooth_ckpt,
            None,  # lora path
            self.lora_alpha,
        )
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.img2video_generator  = torch.Generator(device=device)
        self.img2video_generator.manual_seed(self.i2v_config.generate.global_seed)
        self.sim_ranges = self.i2v_config.validation_data.mask_sim_range

        # Drag GAN
        self.drag_model = Renderer(disable_timing=True)
    
    def generate_image(self, image, text):
        if text is not None:
            pos_prompt, neg_prompt = text
            self.img2img_model.prepare(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=50,
                guidance_scale=self.guidance_scale,
                delta=self.delta,
            )
        sampled_inputs = [image]
        input_batch = torch.cat(sampled_inputs)
        output_images = self.img2img_model.stream(
            input_batch.to(device=self.img2img_model.device, dtype=self.img2img_model.dtype)
        ).cpu()
        return output_images
    
    def generate_video(self, image, text, height=None, width=None):
        pos_prompt, neg_prompt = text 
        sim_range = self.sim_ranges[0]
        print(f"using sim_range : {sim_range}")
        self.i2v_config.validation_data.mask_sim_range = sim_range
        sample = self.img2video_model(
            image = image,
            prompt = pos_prompt,
            generator       = self.img2video_generator,
            video_length    = self.i2v_config.generate.video_length,
            height          = height if height is not None else self.i2v_config.generate.sample_height,
            width           = width if width is not None else self.i2v_config.generate.sample_width,
            negative_prompt = neg_prompt,
            mask_sim_template_idx = self.i2v_config.validation_data.mask_sim_range,
            **self.i2v_config.validation_data,
        ).videos
        return sample
    
    def prepare_drag_model(
        self,
        custom_image: Image,
        latent_space = 'w+',
        trunc_psi = 0.7,
        trunc_cutoff = None,
        seed = 0,
        lr = 0.001,
        generator_params = dnnlib.EasyDict(),
        pretrained_weight = 'stylegan2_lions_512_pytorch',
    ):
        self.drag_model.init_network(
            generator_params,  # res
            pretrained_weight,  # pkl
            seed,  # w0_seed,
            None,  # w_load
            latent_space == 'w+',  # w_plus
            'const',
            trunc_psi,  # trunc_psi,
            trunc_cutoff,  # trunc_cutoff,
            None,  # input_transform
            lr  # lr,
        )

        percept = PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=True
        )

        pti = PTI(self.drag_model.G, percept, max_pti_step=400)
        inversed_img, w_pivot = pti.train(custom_image, latent_space == 'w+')
        inversed_img = (inversed_img[0] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        inversed_img = inversed_img.cpu().numpy()
        inversed_img = Image.fromarray(inversed_img)
        mask = np.ones((inversed_img.height, inversed_img.width),
                                    dtype=np.uint8)
        generator_params.image = inversed_img
        generator_params.w = w_pivot.detach().cpu().numpy()
        self.drag_model.set_latent(w_pivot, trunc_psi, trunc_cutoff)

        del percept
        del pti
        print('inverse end')

        return generator_params, mask

    def drag_image(
        self,
        points,
        mask,
        motion_lambda = 20,
        r1_in_pixels = 3,
        r2_in_pixels = 12,
        trunc_psi = 0.7,
        draw_interval = 1,
        generator_params = dnnlib.EasyDict(),
    ):
        p_in_pixels = []
        t_in_pixels = []
        valid_points = []
        # Transform the points into torch tensors
        for key_point, point in points.items():
            try:
                p_start = point.get("start_temp", point["start"])
                p_end = point["target"]

                if p_start is None or p_end is None:
                    continue

            except KeyError:
                continue

            p_in_pixels.append(p_start)
            t_in_pixels.append(p_end)
            valid_points.append(key_point)

        mask = torch.tensor(mask).float()
        drag_mask = 1 - mask

        # reverse points order
        p_to_opt = reverse_point_pairs(p_in_pixels)
        t_to_opt = reverse_point_pairs(t_in_pixels)
        step_idx = 0

        self.drag_model._render_drag_impl(
            generator_params,
            p_to_opt,  # point
            t_to_opt,  # target
            drag_mask,  # mask,
            motion_lambda,  # lambda_mask
            reg = 0,
            feature_idx = 5,  # NOTE: do not support change for now
            r1 = r1_in_pixels,  # r1
            r2 = r2_in_pixels,  # r2
            # random_seed     = 0,
            # noise_mode      = 'const',
            trunc_psi = trunc_psi,
            # force_fp32      = False,
            # layer_name      = None,
            # sel_channels    = 3,
            # base_channel    = 0,
            # img_scale_db    = 0,
            # img_normalize   = False,
            # untransform     = False,
            is_drag=True,
            to_pil=True
        )


        points_upd = points
        if step_idx % draw_interval == 0:
            for key_point, p_i, t_i in zip(valid_points, p_to_opt,
                                            t_to_opt):
                points_upd[key_point]["start_temp"] = [
                    p_i[1],
                    p_i[0],
                ]
                points_upd[key_point]["target"] = [
                    t_i[1],
                    t_i[0],
                ]
                start_temp = points_upd[key_point][
                    "start_temp"]

        image_result = generator_params['image']

        return image_result

# ============================= Model Implementation ENd ===================================


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true',default='True')
parser.add_argument('--cache-dir', type=str, default='./checkpoints')
parser.add_argument(
    "--listen",
    action="store_true",
    help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests",
)
args = parser.parse_args()


class CustomImageMask(gr.Image):
    is_template = True
    def __init__(
        self,
        source='upload', 
        tool='sketch', 
        elem_id="image_upload", 
        label='Generated Image', 
        type="pil", 
        mask_opacity=0.5, 
        brush_color='#FFFFFF', 
        height=400, 
        interactive=True,
        **kwargs
    ):
        super(CustomImageMask, self).__init__(
            source=source, 
            tool=tool, 
            elem_id=elem_id, 
            label=label, 
            type=type, 
            mask_opacity=mask_opacity, 
            brush_color=brush_color, 
            height=height, 
            interactive=interactive,
            **kwargs
        )

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == 'sketch' and self.source in ['upload', 'webcam'] and type(x) != dict:
            decode_image = gr.processing_utils.decode_base64_to_image(x)
            width, height = decode_image.size
            mask = np.ones((height, width, 4), dtype=np.uint8)
            mask[..., -1] = 255
            mask = self.postprocess(mask)
            x = {'image': x, 'mask': mask}
        # decode_image = gr.processing_utils.decode_base64_to_image(x['image'])
        # width, height = decode_image.size
        # decode_mask = np.ones((width, height, 4), dtype=np.uint8)
        # decode_mask[..., -1] = 255
        # x = {
        #     'image': self.postprocess(decode_image),
        #     'mask': self.postprocess(decode_mask)
        # }
        return super().preprocess(x)


draggan_ckpts = os.listdir('checkpoints/drag')
draggan_ckpts.sort()


generate_pipeline = GeneratePipeline()
generate_pipeline.init_model()


with gr.Blocks() as demo:
    global_state = gr.State(
        {
            'is_image_generation': True,
            'is_image_text_prompt_up-to-date': True,
            'is_show_mask': False,
            'is_dragging': False,
            'generated_image': None,
            'generated_image_show': None,
            'drag_markers': [
                {
                    'points': {},
                    'mask': None
                }
            ],
            'generator_params': dnnlib.EasyDict(),
            'default_image_text_prompts': ('', 'low quality, bad quality, blurry, low resolution'),
            'default_video_text_prompts': ('', 'wrong white balance, dark, sketches,worst quality,low quality, deformed, distorted, disfigured, bad eyes, wrong lips,weird mouth, bad teeth, mutated hands and fingers, bad anatomy,wrong anatomy, amputation, extra limb, missing limb, floating,limbs, disconnected limbs, mutation, ugly, disgusting, bad_pictures, negative_hand-neg'),
            'image_text_prompts': ('', 'low quality, bad quality, blurry, low resolution'),
            'video_text_prompts': ('', 'wrong white balance, dark, sketches,worst quality,low quality, deformed, distorted, disfigured, bad eyes, wrong lips,weird mouth, bad teeth, mutated hands and fingers, bad anatomy,wrong anatomy, amputation, extra limb, missing limb, floating,limbs, disconnected limbs, mutation, ugly, disgusting, bad_pictures, negative_hand-neg'),
            'params': {
                'seed': 0,
                'motion_lambda': 20,
                'r1_in_pixels': 3,
                'r2_in_pixels': 12,
                'magnitude_direction_in_pixels': 1.0,
                'latent_space': 'w+',
                'trunc_psi': 0.7,
                'trunc_cutoff': None,
                'lr': 0.001,
            },
            'device': None, # device,
            'draw_interval': 1,
            'points': {},
            'curr_point': None,
            'curr_type_point': 'start',
            'editing_state': 'add_points',
            'pretrained_weight': draggan_ckpts[0],
            'video_preview_resolution': '512 x 512',
            'viewer_height': 300,
            'viewer_width': 300
        }
    )

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=8, min_width=10):
                with gr.Tab('Image Text Prompts'):
                    image_pos_text_prompt_editor = gr.Textbox(placeholder='Positive Prompts', label='Positive', min_width=10)
                    image_neg_text_prompt_editor = gr.Textbox(placeholder='Negative Prompts', label='Negative', min_width=10)
                with gr.Tab('Video Text Prompts'):
                    video_pos_text_prompt_editor = gr.Textbox(placeholder='Positive Prompts', label='Positive', min_width=10)
                    video_neg_text_prompt_editor = gr.Textbox(placeholder='Negative Prompts', label='Negative', min_width=10)
                with gr.Tab('Drag Image'):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=10):
                            drag_mode_on_button = gr.Button('Drag Mode On', size='sm', min_width=10)
                            drag_mode_off_button = gr.Button('Drag Mode Off', size='sm', min_width=10)
                            drag_checkpoint_dropdown = gr.Dropdown(choices=draggan_ckpts, value=draggan_ckpts[0], label='checkpoint', min_width=10)
                        with gr.Column(scale=1, min_width=10):
                            with gr.Row():
                                drag_start_button = gr.Button('start', size='sm', min_width=10)
                                drag_stop_button = gr.Button('stop', size='sm', min_width=10)
                            with gr.Row():
                                add_point_button = gr.Button('add point', size='sm', min_width=10)
                                reset_point_button = gr.Button('reset point', size='sm', min_width=10)
                            with gr.Row():
                                steps_number = gr.Number(0, label='steps', interactive=False)
                        with gr.Column(scale=1, min_width=10):
                            with gr.Row():
                                draw_mask_button = gr.Button('draw mask', size='sm', min_width=10)
                                reset_mask_button = gr.Button('reset mask', size='sm', min_width=10)
                            with gr.Row():
                                show_mask_checkbox = gr.Checkbox(value=False, label='show mask', min_width=10, interactive=True)
                            # with gr.Row():
                            #     gr.Button('flexible area', size='sm', min_width=10)
                            #     gr.Button('fixed area', size='sm', min_width=10)
                            # with gr.Row(equal_height=False):
                            #     gr.Button('reset mask', size='sm', min_width=10)
                            #     gr.Checkbox(label='show mask', min_width=10)
                            # with gr.Row():
                            with gr.Row():
                                motion_lambda_number = gr.Number(20, label='Motion Lambda', minimum=1, maximum=100, step=1, interactive=True)
                with gr.Tab('Settings'):
                    with gr.Row():
                        # with gr.Column(scale=1, min_width=10):
                        #     gr.Button('Export Captured Image', size='sm', min_width=10)
                        #     gr.Button('Export Generated Image', size='sm', min_width=10)
                        #     gr.Button('Export Generated Video', size='sm', min_width=10) 
                        with gr.Column(scale=2, min_width=10):
                            video_preview_resolution_dropdown = gr.Dropdown(choices=['256 x 256', '512 x 512'], value='512 x 512', label='Video Preview Resolution', min_width=10)
                            # gr.Dropdown(['256 x 256', '512 x 512'], label='Video Export Resolution', min_width=10)
            with gr.Column(scale=1, min_width=10):
                confirm_text_button = gr.Button('Confirm Text', size='sm', min_width=10)
                generate_video_button = gr.Button('Generate Video', size='sm', min_width=10)
                clear_video_button = gr.Button('Clear Video', size='sm', min_width=10)
        with gr.Row():
            captured_image_viewer = gr.Image(source='upload', tool='color-sketch', type='pil', label='Image Drawer', height=global_state.value['viewer_height'], width=global_state.value['viewer_width'], interactive=True, shape=(global_state.value['viewer_width'], global_state.value['viewer_height']))  # 
            generated_image_viewer = CustomImageMask(source='upload', tool='sketch', elem_id="image_upload", label='Generated Image', type="pil", mask_opacity=0.5, brush_color='#FFFFFF', height=global_state.value['viewer_height'], width=global_state.value['viewer_width'], interactive=True)
            generated_video_viewer = gr.Video(source='upload', label='Generated Video', height=global_state.value['viewer_height'], width=global_state.value['viewer_width'], interactive=False)
            # generated_video_viewer = gr.Image(source='upload', label='Generated Video', interactive=False)
    
    gr.Markdown(
        """
            ## Quick Start

            1. Select one sample image in `More` tab.
            2. Draw to edit the sample image in the left most image viewer.
            3. Click `Generate Video` and enjoy it!

            ## Advance Usage

            1. **Try different text prompts.** Enter positive or negative prompts for image / video generation, and
            click `Confirm Text` to enable your prompts.
            2. **Drag images.** Go to `Drag Image` tab, choose a suitable checkpoint and click `Drag Mode On`. 
            It might take a minute to prepare. Properly add points and use masks, then click `start` to 
            start dragging. Once you think it's ok, click `stop` button.
            3. **Adjust video resolution** in the `More` tab.
            4. **Draw from scratch** by choosing `canvas.jpg` in `More` tab and enjoy yourself!
        """
    )
    
    # ========================= Main Function Start =============================
    def on_captured_image_viewer_update(state, image):
        if image is None:
            return state, gr.Image.update(None)
        if state['is_image_text_prompt_up-to-date']:
            text_prompts = None
        else:
            text_prompts = state['image_text_prompts']
        state['is_image_text_prompt_up-to-date'] = True

        input_image = np.array(image).astype(np.float32)
        input_image = (input_image / 255 - 0.5) * 2
        input_image = torch.tensor(input_image).permute([2, 0, 1])
        noisy_image = torch.randn_like(input_image)
        output_image = generate_pipeline.generate_image(
            input_image,
            text_prompts,
        )[0]
        output_image = generate_pipeline.generate_image(
            noisy_image,
            None,
        )[0]  # TODO: is there more elegant way?
        output_image = output_image.permute([1, 2, 0])
        output_image = (output_image / 2 + 0.5).clamp(0, 1) * 255
        
        output_image = output_image.to(torch.uint8).cpu().numpy()
        output_image = Image.fromarray(output_image)

        # output_image = image
        state['generated_image'] = output_image
        output_image = update_state_image(state)
        return state, gr.Image.update(output_image, interactive=False)

    captured_image_viewer.change(
        fn=on_captured_image_viewer_update,
        inputs=[global_state, captured_image_viewer], 
        outputs=[global_state, generated_image_viewer]
    )

    def on_generated_image_viewer_edit(state, data_dict):
        mask = data_dict['mask']
        state['drag_markers'][0]['mask'] = np.array(mask)[:, :, 0] // 255
        image = update_state_image(state)
        return state, image
    
    generated_image_viewer.edit(
        fn=on_generated_image_viewer_edit, 
        inputs=[global_state, generated_image_viewer], 
        outputs=[global_state, generated_image_viewer]
    )

    def on_generate_video_click(state):
        input_image = np.array(state['generated_image'])
        text_prompts = state['video_text_prompts']
        video_preview_resolution = state['video_preview_resolution'].split('x')
        height = int(video_preview_resolution[0].strip(' '))
        width = int(video_preview_resolution[1].strip(' '))
        output_video = generate_pipeline.generate_video(
            input_image,
            text_prompts,
            height = height,
            width = width
        )[0]
        output_video = output_video.clamp(0, 1) * 255
        output_video = output_video.to(torch.uint8)
        # 3 T H W
        print('[video generation done]')
        
        fps = 5  # frames per second
        video_size = (height, width)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('results/gradio_temp.mp4', fourcc, fps, video_size)  # Create VideoWriter object

        for i in range(output_video.shape[1]):
            frame = output_video[:, i, :, :].permute([1, 2, 0]).cpu().numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_writer.write(frame)

        video_writer.release()
        return state, gr.Video.update('results/gradio_temp.mp4')
    
    generate_video_button.click(
        fn=on_generate_video_click,
        inputs=[global_state],
        outputs=[global_state, generated_video_viewer]
    )

    def on_clear_video_click(state):
        return state, gr.Video.update(None)
    
    clear_video_button.click(
        fn=on_clear_video_click,
        inputs=[global_state],
        outputs=[global_state, generated_video_viewer]
    )

    def on_drag_mode_on_click(state):
        # prepare DragGAN for custom image
        custom_image = state['generated_image']
        current_ckpt_name = state['pretrained_weight']
        generate_pipeline.prepare_drag_model(
            custom_image,
            generator_params = state['generator_params'],
            pretrained_weight = os.path.join('checkpoints/drag/', current_ckpt_name),
        )
        state['generated_image'] = state['generator_params'].image
        view_image = update_state_image(state)
        return state, gr.Image.update(view_image, interactive=True)
    
    drag_mode_on_button.click(
        fn=on_drag_mode_on_click,
        inputs=[global_state],
        outputs=[global_state, generated_image_viewer]
    )

    def on_drag_mode_off_click(state, image):
        return on_captured_image_viewer_update(state, image)
    
    drag_mode_off_button.click(
        fn=on_drag_mode_off_click,
        inputs=[global_state, captured_image_viewer],
        outputs=[global_state, generated_image_viewer]
    )

    def on_drag_start_click(state):
        state['is_dragging'] = True
        points = state['drag_markers'][0]['points']
        if state['drag_markers'][0]['mask'] is None:
            mask = np.ones((state['generator_params'].image.height, state['generator_params'].image.width), dtype=np.uint8)
        else:
            mask = state['drag_markers'][0]['mask']
        cur_step = 0
        while True:
            if not state['is_dragging']:
                break
            generated_image = generate_pipeline.drag_image(
                points,
                mask,
                motion_lambda = state['params']['motion_lambda'],
                generator_params = state['generator_params']
            )
            state['drag_markers'] = [{'points': points, 'mask': mask}]
            state['generated_image'] = generated_image
            cur_step += 1
            view_image = update_state_image(state)
            if cur_step % 50 == 0:
                print('[{} / {}]'.format(cur_step, 'inf'))
            yield (
                state,
                gr.Image.update(view_image, interactive=False),  # generated image viewer
                gr.Number.update(cur_step),  # step
                # gr.Image.update(interactive=False),  # captured image viewer
                # gr.Button.update(interactive=False),  # confirm text button
                # gr.Button.update(interactive=False),  # generate video button
                # gr.Button.update(interactive=False),  # drag mode on button
                # gr.Button.update(interactive=False),  # drag mode off button
                # gr.Button.update(interactive=False),  # drag start button
                # gr.Button.update(interactive=False),  # draw mask button
                # gr.Button.update(interactive=False),  # reset mask button
                # gr.Button.update(interactive=False),  # add point button
                # gr.Button.update(interactive=False),  # reset point button
                # gr.Checkbox.update(interactive=False),  # show mask checkbox
            )
        
        view_image = update_state_image(state)
        return (
            state, 
            gr.Image.update(view_image, interactive=True), 
            gr.Number.update(cur_step),
            # gr.Image.update(interactive=True),
            # gr.Button.update(interactive=True),
            # gr.Button.update(interactive=True),
            # gr.Button.update(interactive=True),
            # gr.Button.update(interactive=True),
            # gr.Button.update(interactive=True),
            # gr.Button.update(interactive=True),
            # gr.Button.update(interactive=True),
            # gr.Button.update(interactive=True),
            # gr.Button.update(interactive=True),
            # gr.Checkbox.update(interactive=True)
        )
    
    drag_start_button.click(
        fn=on_drag_start_click,
        inputs=[global_state],
        outputs=[global_state, generated_image_viewer, steps_number]
        # , captured_image_viewer, confirm_text_button, generate_video_button, 
        #          drag_mode_on_button, drag_mode_off_button,
        #          drag_start_button, draw_mask_button, reset_mask_button,
        #          add_point_button, reset_point_button, 
        #          show_mask_checkbox
    )

    def on_drag_stop_click(state):
        state['is_dragging'] = False
        return state
    
    drag_stop_button.click(
        fn=on_drag_stop_click,
        inputs=[global_state],
        outputs=[global_state]
    )

    # ========================= Main Function End =============================

    # ====================== Update Text Prompts Start ====================
    def on_image_pos_text_prompt_editor_submit(state, text):
        if len(text) == 0:
            temp = state['image_text_prompts']
            state['image_text_prompts'] = (state['default_image_text_prompts'][0], temp[1])
        else:
            temp = state['image_text_prompts']
            state['image_text_prompts'] = (text, temp[1])
        state['is_image_text_prompt_up-to-date'] = False
        return state
    
    image_pos_text_prompt_editor.submit(
        fn=on_image_pos_text_prompt_editor_submit, 
        inputs=[global_state, image_pos_text_prompt_editor], 
        outputs=None
    )

    def on_image_neg_text_prompt_editor_submit(state, text):
        if len(text) == 0:
            temp = state['image_text_prompts']
            state['image_text_prompts'] = (temp[0], state['default_image_text_prompts'][1])
        else:
            temp = state['image_text_prompts']
            state['image_text_prompts'] = (temp[0], text)
        state['is_image_text_prompt_up-to-date'] = False
        return state
    
    image_neg_text_prompt_editor.submit(
        fn=on_image_neg_text_prompt_editor_submit, 
        inputs=[global_state, image_neg_text_prompt_editor], 
        outputs=None
    )

    def on_video_pos_text_prompt_editor_submit(state, text):
        if len(text) == 0:
            temp = state['video_text_prompts']
            state['video_text_prompts'] = (state['default_video_text_prompts'][0], temp[1])
        else:
            temp = state['video_text_prompts']
            state['video_text_prompts'] = (text, temp[1])
        return state
    
    video_pos_text_prompt_editor.submit(
        fn=on_video_pos_text_prompt_editor_submit, 
        inputs=[global_state, video_pos_text_prompt_editor], 
        outputs=None
    )

    def on_video_neg_text_prompt_editor_submit(state, text):
        if len(text) == 0:
            temp = state['video_text_prompts']
            state['video_text_prompts'] = (temp[0], state['default_video_text_prompts'][1])
        else:
            temp = state['video_text_prompts']
            state['video_text_prompts'] = (temp[0], text)
        return state
    
    video_neg_text_prompt_editor.submit(
        fn=on_video_neg_text_prompt_editor_submit, 
        inputs=[global_state, video_neg_text_prompt_editor], 
        outputs=None
    )

    def on_confirm_text_click(state, image, img_pos_t, img_neg_t, vid_pos_t, vid_neg_t):
        state = on_image_pos_text_prompt_editor_submit(state, img_pos_t)
        state = on_image_neg_text_prompt_editor_submit(state, img_neg_t)
        state = on_video_pos_text_prompt_editor_submit(state, vid_pos_t)
        state = on_video_neg_text_prompt_editor_submit(state, vid_neg_t)
        return on_captured_image_viewer_update(state, image)
    
    confirm_text_button.click(
        fn=on_confirm_text_click,
        inputs=[global_state, captured_image_viewer, image_pos_text_prompt_editor, image_neg_text_prompt_editor,
                video_pos_text_prompt_editor, video_neg_text_prompt_editor],
        outputs=[global_state, generated_image_viewer]
    )

    # ====================== Update Text Prompts End ====================

    # ======================= Drag Point Edit Start =========================

    def on_image_clicked(state, evt: gr.SelectData):
        """
            This function only support click for point selection
        """
        pos_x, pos_y = evt.index
        drag_markers = state['drag_markers']
        key_points = list(drag_markers[0]['points'].keys())
        key_points.sort(reverse=False)
        if len(key_points) == 0:  # no point pairs, add a new point pair
            drag_markers[0]['points'][0] = {
                'start_temp': [pos_x, pos_y],
                'start': [pos_x, pos_y],
                'target': None,
            }
        else:
            largest_id = key_points[-1]
            if drag_markers[0]['points'][largest_id]['target'] is None:  # target is not set
                drag_markers[0]['points'][largest_id]['target'] = [pos_x, pos_y]
            else:  # target is set, add a new point pair
                drag_markers[0]['points'][largest_id + 1] = {
                    'start_temp': [pos_x, pos_y],
                    'start': [pos_x, pos_y],
                    'target': None,
                }
        state['drag_markers'] = drag_markers
        image = update_state_image(state)
        return state, gr.Image.update(image, interactive=False)
    
    generated_image_viewer.select(
        fn=on_image_clicked,
        inputs=[global_state],
        outputs=[global_state, generated_image_viewer],
    )

    def on_add_point_click(state):
        return gr.Image.update(state['generated_image_show'], interactive=False)
    
    add_point_button.click(
        fn=on_add_point_click,
        inputs=[global_state],
        outputs=[generated_image_viewer]
    )

    def on_reset_point_click(state):
        drag_markers = state['drag_markers']
        drag_markers[0]['points'] = {}
        state['drag_markers'] = drag_markers
        image = update_state_image(state)
        return state, gr.Image.update(image)
    
    reset_point_button.click(
        fn=on_reset_point_click,
        inputs=[global_state],
        outputs=[global_state, generated_image_viewer]
    )

    # ======================= Drag Point Edit End =========================

    # ======================= Drag Mask Edit Start =========================

    def on_draw_mask_click(state):
        return gr.Image.update(state['generated_image_show'], interactive=True)
    
    draw_mask_button.click(
        fn=on_draw_mask_click,
        inputs=[global_state],
        outputs=[generated_image_viewer]
    )

    def on_reset_mask_click(state):
        drag_markers = state['drag_markers']
        drag_markers[0]['mask'] = np.ones_like(drag_markers[0]['mask'])
        state['drag_markers'] = drag_markers
        image = update_state_image(state)
        return state, gr.Image.update(image)
    
    reset_mask_button.click(
        fn=on_reset_mask_click,
        inputs=[global_state],
        outputs=[global_state, generated_image_viewer]
    )

    def on_show_mask_click(state, evt: gr.SelectData):
        state['is_show_mask'] = evt.selected
        image = update_state_image(state)
        return state, image

    show_mask_checkbox.select(
        fn=on_show_mask_click,
        inputs=[global_state],
        outputs=[global_state, generated_image_viewer]
    )

    # ======================= Drag Mask Edit End =========================

    # ======================= Drag Setting Start =========================

    def on_motion_lambda_change(state, number):
        state['params']['number'] = number
        return state
    
    motion_lambda_number.input(
        fn=on_motion_lambda_change,
        inputs=[global_state, motion_lambda_number],
        outputs=[global_state]
    )

    def on_drag_checkpoint_change(state, checkpoint):
        state['pretrained_weight'] = checkpoint
        print(type(checkpoint), checkpoint)
        return state
    
    drag_checkpoint_dropdown.change(
        fn=on_drag_checkpoint_change,
        inputs=[global_state, drag_checkpoint_dropdown],
        outputs=[global_state]
    )

    # ======================= Drag Setting End =========================

    # ======================= General Setting Start =========================

    def on_video_preview_resolution_change(state, resolution):
        state['video_preview_resolution'] = resolution
        return state
    
    video_preview_resolution_dropdown.change(
        fn=on_video_preview_resolution_change,
        inputs=[global_state, video_preview_resolution_dropdown],
        outputs=[global_state]
    )

    # ======================= General Setting End =========================


demo.queue(concurrency_count=3, max_size=20)
# demo.launch(share=False, server_name="0.0.0.0" if args.listen else "127.0.0.1")
demo.launch()
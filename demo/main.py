import sys, os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from typing import List, Literal, Dict, Optional
import torch
import numpy as np
from PIL import Image
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.streamdiffusion.wrapper import StreamDiffusionWrapper

from models.animatediff.pipelines import I2VPipeline
from models.animatediff.utils.util import save_videos_grid
from omegaconf import OmegaConf

from PyQt5.QtCore import Qt, QRect, QTimer, QObject, QEvent
from PyQt5.QtCore import QThread, QMutex
from PyQt5.QtGui import QPixmap, QIcon, QImage, QRegion
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from Ui_frame import Ui_MainWindow

from models.draggan.viz.renderer import Renderer
from models.draggan.gan_inv.lpips.util import PerceptualLoss
import models.draggan.dnnlib as dnnlib
from models.draggan.gan_inv.inversion import PTI

from draw_utils import draw_points_on_image, draw_mask_on_image, draw_circle_on_mask
import imageio
import torchvision
from einops import rearrange


# =================== AUTO SAVE SETTINGS ==================
CAPTURE_AUTO_SAVE = False  # left viewer
IGEN_AUTO_SAVE = False  # middle viewer
VGEN_AUTO_SAVE = False  # right viewer
# If you want the demo to automatically save results, please
# set corresponding constants to "True".
# Defaultly, results will be saved to "results/".
# =================== AUTO SAVE SETTINGS ==================


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


def pixmap2array(pixmap):
    image = pixmap.toImage()
    width = image.width()
    height = image.height()
    array = np.frombuffer(image.bits().asstring(width * height * 4), dtype=np.uint8).reshape((height, width, 4))
    array = np.stack([
        array[:, :, 2],
        array[:, :, 1],
        array[:, :, 0],
        array[:, :, 3],
    ], axis=2)  # seemingly image.bits() is saved in the order "BGRA"
    return array


def pixmap2tensor(pixmap):
    image = pixmap.toImage()
    width = image.width()
    height = image.height()
    array = np.frombuffer(image.bits().asstring(width * height * 4), dtype=np.uint8).reshape((height, width, 4))
    array = np.stack([
        array[:, :, 2],
        array[:, :, 1],
        array[:, :, 0],
        array[:, :, 3],
    ], axis=2)  # seemingly image.bits() is saved in the order "BGRA"
    return torch.tensor(array)


def array2qimg(array):
    height, width, channel = array.shape
    bytesPerLine = 4 * width
    qImg = QImage(array.data, width, height, bytesPerLine, QImage.Format_RGBA8888)
    return qImg


def tensor2qimg(tensor):
    array = tensor.cpu().numpy()
    height, width, channel = array.shape
    bytesPerLine = 4 * width
    qImg = QImage(array.data, width, height, bytesPerLine, QImage.Format_RGBA8888)
    return qImg


def render_view_image(piximg, drag_markers, show_mask=False):
    img = pixmap2array(piximg)
    img = Image.fromarray(img)
    img = draw_points_on_image(img, drag_markers['points'])
    if show_mask:
        img = draw_mask_on_image(img, drag_markers['mask'])
    img = np.array(img).astype(np.uint8)
    img = np.concatenate([
        img,
        255 * np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
    ], axis=2)
    return QPixmap.fromImage(array2qimg(img))


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
        self.img2video_generator  = torch.Generator(device='cuda')
        self.img2video_generator.manual_seed(self.i2v_config.generate.global_seed)
        self.sim_ranges = self.i2v_config.validation_data.mask_sim_range

        # Drag GAN
        self.drag_model = Renderer(disable_timing=True)
    
    def generate_image(self, image, text):
        if text is not None:
            pos_prompt, neg_prompt = text
            # print('i2i prepare start')
            # torch.cuda.synchronize()
            # start_time = time.perf_counter()
            self.img2img_model.prepare(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=50,
                guidance_scale=self.guidance_scale,
                delta=self.delta,
            )
            # torch.cuda.synchronize()
            # end_time = time.perf_counter()
            # print('i2i prepare end: {:.6f}'.format(end_time-start_time))
        sampled_inputs = [image]
        input_batch = torch.cat(sampled_inputs)
        # print('i2i generation start')
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        output_images = self.img2img_model.stream(
            input_batch.to(device=self.img2img_model.device, dtype=self.img2img_model.dtype)
        ).cpu()
        # torch.cuda.synchronize()
        # end_time = time.perf_counter()
        # print('i2i generation end: {:.6f}'.format(end_time-start_time))
        return output_images
    
    def generate_video(self, image, text, height=None, width=None):
        pos_prompt, neg_prompt = text 
        sim_range = self.sim_ranges[0]
        print(f"using sim_range : {sim_range}")
        self.i2v_config.validation_data.mask_sim_range = sim_range
        # print('i2v generation start')
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
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
        # torch.cuda.synchronize()
        # end_time = time.perf_counter()
        # print('i2v generation end: {:.6f}'.format(end_time-start_time))
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
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
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
        # torch.cuda.synchronize()
        # end_time = time.perf_counter()
        # print('drag prepare end: {:.6f}'.format(end_time-start_time))
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


INIT_MODES = ['init']
NORMAL_MODES = ['normal']
DRAG_MODES = ['drag_add_point', 'drag_paint_flexible_mask', 'drag_paint_fixed_mask']
DRAG_ADD_POINT_MODE = 'drag_add_point'
DRAG_PAINT_MASK_MODES = ['drag_paint_flexible_mask', 'drag_paint_fixed_mask']
DRAG_PAINT_FLEX_MASK_MODE = 'drag_paint_flexible_mask'
DRAG_PAINT_FIX_MASK_MODE = 'drag_paint_fixed_mask'
NORMAL_MODE = 'normal'
INIT_MODE = 'init'


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("frame")  # set title
        self.setWindowIcon(QIcon("logo.ico"))  # add icon
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)  # on the top and remove frame
        self.ui.exitButton.clicked.connect(self.on_exit_click)  # exit the app
        self.ui.generateVideoButton.clicked.connect(self.on_generate_video_click)  # start to generate video
        self.ui.clearVideoButton.clicked.connect(self.on_clear_video_click)  # clear the video generated
        self.ui.confirmTextPromptButton.clicked.connect(self.on_confirm_text_prompt_click)  # confirm text prompts
        # TODO: when prompts is not changed, do nothing
        self.ui.dragModeOnButton.clicked.connect(self.on_drag_mode_on_click)  # open the drag mode will lock actions for "Capture", "Image Generation"
        self.ui.dragModeOffButton.clicked.connect(self.on_drag_mode_off_click)  # close the drag mode will make "Capture", "Image Generation" work again
        self.ui.dragStartButton.clicked.connect(self.on_drag_start_click)  # start drag generation
        self.ui.dragStopButton.clicked.connect(self.on_drag_stop_click)  # stop drag generation
        self.ui.dragAddPointButton.clicked.connect(self.on_drag_add_point_click)  # start to add points
        self.ui.dragResetPointButton.clicked.connect(self.on_drag_reset_point_click)  # clear the key points for DragGAN
        self.ui.dragShowMaskCheck.stateChanged.connect(self.on_drag_show_mask_changed)  # change the show mask signal
        self.ui.dragResetMaskButton.clicked.connect(self.on_drag_reset_mask_clicked)  # clear the mask for DragGAN
        self.ui.dragFixedAreaButton.clicked.connect(self.on_drag_fixed_area_clicked)
        self.ui.dragFlexibleAreaButton.clicked.connect(self.on_drag_flexible_area_clicked)
        self.ui.dragMaskBrushRadiusAddButton.clicked.connect(self.on_drag_mask_brush_radius_add_clicked)
        self.ui.dragMaskBrushRadiusMinusButton.clicked.connect(self.on_drag_mask_brush_radius_minus_clicked)
        self.ui.dragMotionLambdaAddButton.clicked.connect(self.on_drag_motion_lambda_add_clicked)
        self.ui.dragMotionLambdaMinusButton.clicked.connect(self.on_drag_motion_lambda_minus_clicked)
        self.ui.exportCapturedImageButton.clicked.connect(self.on_export_captured_image_clicked)
        self.ui.exportGeneratedImageButton.clicked.connect(self.on_export_generated_image_clicked)
        self.ui.exportGeneratedVideoButton.clicked.connect(self.on_export_generated_video_clicked)
        self.ui.confirmCheckpointsButton.clicked.connect(self.on_confirm_checkpoints_clicked)
        self.ui.imgOutput.setScaledContents(True)  # show generated image (real time)
        self.ui.videoOutput.setScaledContents(True)  # show generated video
        self.main_window_event_filter = MainWindowEventFilter(self)
        self.drag_edit_event_filter = DragEditEventFilter(self)
        self.installEventFilter(self.main_window_event_filter)
        self.ui.imgOutput.installEventFilter(self.drag_edit_event_filter)  # in the "drag" mode, process some events
        self.ui.hideControllerButton.clicked.connect(self.on_hide_controller_clicked)
        size_hint = self.ui.controller.sizePolicy()
        size_hint.setRetainSizeWhenHidden(True)
        self.ui.controller.setSizePolicy(size_hint)
        size_hint = self.ui.progressBar.sizePolicy()
        size_hint.setRetainSizeWhenHidden(True)
        self.ui.progressBar.setSizePolicy(size_hint)
        self.ui.progressBar.setVisible(False)
        self.is_exit = False  # signal whether the "exit" is clicked, terminate sub threads
        self.is_generate_video = False  # signal whether video generation is running
        self.is_video_uptodate = True  # signal whether video player parameters (in "Video Viewer") is up-to-date
        # this signal is set "False" when a new video generation is done, and set "True" when "Video Viewer" reset player parameters
        self.mode = 'normal'  # ['normal', 'drag_add_point', 'drag_paint_flexible_mask', 'drag_paint_fixed_mask']
        self.default_image_text_prompts = ("", "low quality, bad quality, blurry, low resolution")
        # (positive, negative)
        self.default_video_text_prompts = ("", "wrong white balance, dark, sketches,worst quality,low quality, deformed, distorted, disfigured, bad eyes, wrong lips,weird mouth, bad teeth, mutated hands and fingers, bad anatomy,wrong anatomy, amputation, extra limb, missing limb, floating,limbs, disconnected limbs, mutation, ugly, disgusting, bad_pictures, negative_hand-neg")
        # (positive, negative)

        self.captured_images = []  # save captured images, currently only the last image
        self.generated_images = []  # save generated images, currently only the last image
        # tips: both image generated by StreamDiffusion and DragGAN
        self.generated_video_images = []  # save generated video images, currently only frames of the last video
        self.drag_markers = [
            {
                'points': {},
                'mask': None
            }
        ]  # save drag markers, currently only the last image
        # i.e. points and masks

        self.image_text_prompts = self.default_image_text_prompts  # image text prompts
        self.video_text_prompts = self.default_video_text_prompts  # video text prompts
        self.is_image_text_prompt_uptodate = True  # signal whether text prompts used are up-to-date
        # in StreamDiffusion, if the signal is "False", requires a .prepare() call to restart the pipeline.

        self.is_generate_drag_image = False  # signal whether to generate drag image
        # after clicking "start", this signal will be set true
        # after clicking "stop" or drag generation ending, this signal will be set false
        self.generator_params = dnnlib.EasyDict()  # used in DragGAN, save image, weight, etc.
        self.is_show_mask = False  # signal whether to show mask for DragGAN
        self.is_paint_mask = False  # for mask painting
        self.mask_brush_radius = 50
        self.motion_lambda = 20

        self.draggan_checkpoints = self.init_draggan_checkpoints()
        self.i2i_body_checkpoints = self.init_i2i_body_checkpoints()
        self.i2i_lora_checkpoints = self.init_i2i_lora_checkpoints()
        self.i2v_unet_checkpoints = self.init_i2v_unet_checkpoints()
        self.i2v_dreambooth_checkpoints = self.init_i2v_dreambooth_checkpoints()
        self.init_video_export_resolutions()
        self.init_video_preview_resolutions()

        self.is_controller_visible = True

        # --------------- Mutex Definition Start --------------- 
        self.captured_images_mux = QMutex()
        self.generated_images_mux = QMutex()
        self.generated_video_images_mux = QMutex()
        self.is_generate_video_mux = QMutex()
        self.is_video_uptodate_mux = QMutex()
        self.image_text_prompts_mux = QMutex()
        self.video_text_prompts_mux = QMutex()
        self.is_image_text_prompt_uptodate_mux = QMutex()
        self.mode_mux = QMutex()
        self.is_generate_drag_image_mux = QMutex()
        self.drag_markers_mux = QMutex()
        self.generator_params_mux = QMutex()
        self.is_show_mask_mux = QMutex()
        self.mask_brush_radius_mux = QMutex()
        self.motion_lambda_mux = QMutex()
        self.draggan_checkpoints_mux = QMutex()
        self.i2i_body_checkpoints_mux = QMutex()
        self.i2i_lora_checkpoints_mux = QMutex()
        self.i2v_unet_checkpoints_mux = QMutex()
        self.i2v_dreambooth_checkpoints_mux = QMutex()
        self.video_export_resolution_mux = QMutex()
        self.video_preview_resolution_mux = QMutex()
        # --------------- Mutex Definition End --------------- 

        self.capture = Capture(self)  # Capture
        self.capture.start()

        self.image_viewer = GeneratedImageViewer(self)  # Generated Image Viewer
        self.image_viewer.start()
        self.video_viewer = GeneratedVideoViewer(self)  # Generated Video Viewer
        self.video_viewer.start()

        self.generate_pipeline = GeneratePipeline()  # Generate Pipeline
        self.generator_init = GeneratorInit(self)  # initialize generation models
        self.generator_init.start()
        self.generator_init.wait()

        self.image_generation = ImageGeneration(self)  # Image Generation
        self.image_generation.start()

        self.drag_preparing = DragPreparing(self)  # Drag Preparing

        self.drag_generation = DragGeneration(self)  # Drag Generation

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateAlways)
        self.timer.start(50)  # force to update, 20 fps
    
    def paintEvent(self, event):
        # compute the capture region
        capture_region = self.ui.imgCapture.geometry()
        viewer_region = self.ui.viewer.geometry()
        capture_region = QRect(
            capture_region.x() + 3 + viewer_region.x(), 
            capture_region.y() + 3 + viewer_region.y(),
            capture_region.width() - 6,
            capture_region.height() - 6
        )
        # compute mask region
        full_region = QRegion(self.rect(), QRegion.Rectangle)
        capture_region = QRegion(capture_region, QRegion.Rectangle)
        mask_region = full_region.subtracted(capture_region)
        if not self.is_controller_visible:
            controller_region = QRegion(self.ui.controller.rect(), QRegion.Rectangle)
            mask_region = mask_region.subtracted(controller_region)
        self.setMask(mask_region)
    
    def updateAlways(self):
        # Trigger a repaint event at regular intervals
        self.update()
    
    def init_video_export_resolutions(self):
        resolutions = ['256 x 256', '512 x 512']
        default_resolution = '512 x 512'
        self.ui.videoExportResolutionBox.addItems(resolutions)
        self.ui.videoExportResolutionBox.setCurrentText(default_resolution)
    
    def init_video_preview_resolutions(self):
        resolutions = ['256 x 256', '512 x 512']
        default_resolution = '512 x 512'
        self.ui.videoPreviewResolutionBox.addItems(resolutions)
        self.ui.videoPreviewResolutionBox.setCurrentText(default_resolution)
    
    def init_draggan_checkpoints(self):
        dir_name = 'checkpoints/drag'
        checkpoint_names = os.listdir(dir_name)
        checkpoint_names.sort()
        checkpoint_paths = [os.path.join(dir_name, checkpoint_name) for checkpoint_name in checkpoint_names]
        assert len(checkpoint_names) > 0
        draggan_checkpoints = {}
        for checkpoint_name, checkpoint_path in zip(checkpoint_names, checkpoint_paths):
            draggan_checkpoints[checkpoint_name] = checkpoint_path
        self.ui.dragCheckpointBox.addItems(checkpoint_names)
        self.ui.dragCheckpointBox.setCurrentText(checkpoint_names[0])
        return draggan_checkpoints
    
    def init_i2i_body_checkpoints(self):
        dir_name = 'checkpoints/diffusion_body'
        checkpoint_names = os.listdir(dir_name)
        checkpoint_names.sort()
        checkpoint_paths = [os.path.join(dir_name, checkpoint_name) for checkpoint_name in checkpoint_names]
        assert len(checkpoint_names) > 0
        i2i_body_checkpoints = {}
        for checkpoint_name, checkpoint_path in zip(checkpoint_names, checkpoint_paths):
            i2i_body_checkpoints[checkpoint_name] = checkpoint_path
        self.ui.i2iBodyBox.addItems(checkpoint_names)
        self.ui.i2iBodyBox.setCurrentText(checkpoint_names[0])
        return i2i_body_checkpoints
    
    def init_i2i_lora_checkpoints(self):
        dir_name = 'checkpoints/i2i/lora'
        checkpoint_names = os.listdir(dir_name)
        checkpoint_names.sort()
        checkpoint_paths = [os.path.join(dir_name, checkpoint_name) for checkpoint_name in checkpoint_names]
        assert len(checkpoint_names) > 0
        i2i_lora_checkpoints = {}
        for checkpoint_name, checkpoint_path in zip(checkpoint_names, checkpoint_paths):
            i2i_lora_checkpoints[checkpoint_name] = checkpoint_path
        self.ui.i2iLoraBox.addItems(checkpoint_names)
        self.ui.i2iLoraBox.setCurrentText(checkpoint_names[0])
        return i2i_lora_checkpoints
    
    def init_i2v_unet_checkpoints(self):
        dir_name = 'checkpoints/i2v/unet'
        checkpoint_names = os.listdir(dir_name)
        checkpoint_names.sort()
        checkpoint_paths = [os.path.join(dir_name, checkpoint_name) for checkpoint_name in checkpoint_names]
        assert len(checkpoint_names) > 0
        i2v_unet_checkpoints = {}
        for checkpoint_name, checkpoint_path in zip(checkpoint_names, checkpoint_paths):
            i2v_unet_checkpoints[checkpoint_name] = checkpoint_path
        self.ui.i2vUNetBox.addItems(checkpoint_names)
        self.ui.i2vUNetBox.setCurrentText(checkpoint_names[0])
        return i2v_unet_checkpoints
    
    def init_i2v_dreambooth_checkpoints(self):
        dir_name = 'checkpoints/i2v/dreambooth'
        checkpoint_names = os.listdir(dir_name)
        checkpoint_names.sort()
        checkpoint_paths = [os.path.join(dir_name, checkpoint_name) for checkpoint_name in checkpoint_names]
        assert len(checkpoint_names) > 0
        i2v_dreambooth_checkpoints = {}
        for checkpoint_name, checkpoint_path in zip(checkpoint_names, checkpoint_paths):
            i2v_dreambooth_checkpoints[checkpoint_name] = checkpoint_path
        self.ui.i2vDreamboothBox.addItems(checkpoint_names)
        self.ui.i2vDreamboothBox.setCurrentText(checkpoint_names[0])
        return i2v_dreambooth_checkpoints
    
    def on_confirm_checkpoints_clicked(self):
        self.mode_mux.lock()
        self.mode = INIT_MODE
        self.mode_mux.unlock()
        time.sleep(0.2)

        del self.generate_pipeline
        self.generate_pipeline = GeneratePipeline()

        self.i2i_body_checkpoints_mux.lock()
        self.i2i_lora_checkpoints_mux.lock()
        self.i2v_unet_checkpoints_mux.lock()
        self.i2v_dreambooth_checkpoints_mux.lock()

        self.generate_pipeline.i2i_body_ckpt = \
            self.i2i_body_checkpoints[self.ui.i2iBodyBox.currentText()]
        self.generate_pipeline.i2i_lora_dict = {
            self.i2i_lora_checkpoints[self.ui.i2iLoraBox.currentText()]: 1.0
        }

        # self.generate_pipeline.i2v_config = OmegaConf.load('demo/configs/i2v_config.yaml')
        self.generate_pipeline.i2v_config.generate.model_path = \
            self.i2v_unet_checkpoints[self.ui.i2vUNetBox.currentText()]
        self.generate_pipeline.i2v_config.generate.db_path = \
            self.i2v_dreambooth_checkpoints[self.ui.i2vDreamboothBox.currentText()]
        self.generate_pipeline.i2v_unet_path = self.generate_pipeline.i2v_config.generate.model_path
        self.generate_pipeline.i2v_dreambooth_ckpt = self.generate_pipeline.i2v_config.generate.db_path

        self.i2i_body_checkpoints_mux.unlock()
        self.i2i_lora_checkpoints_mux.unlock()
        self.i2v_unet_checkpoints_mux.unlock()
        self.i2v_dreambooth_checkpoints_mux.unlock()

        self.generator_init.start()
        self.generator_init.wait()

        self.mode_mux.lock()
        self.mode = NORMAL_MODE
        self.mode_mux.unlock()

    
    def on_export_captured_image_clicked(self):
        self.captured_images_mux.lock()
        img = Image.fromarray(pixmap2array(self.captured_images[0])[..., :3])
        file_name, _ = QFileDialog.getSaveFileName(
            self, caption='save to file', directory='./', filter='Image (*.jpg)'
        )
        if file_name[-4:] in ['jpg']:
            pass
        else:
            file_name = file_name + '.jpg'
        img.save(file_name)
        self.captured_images_mux.unlock()

    def on_export_generated_image_clicked(self):
        self.generated_images_mux.lock()
        img = Image.fromarray(pixmap2array(self.generated_images[0])[..., :3])
        file_name, _ = QFileDialog.getSaveFileName(
            self, caption='save to file', directory='./', filter='Image (*.jpg)'
        )
        if file_name[-4:] in ['jpg']:
            pass
        else:
            file_name = file_name + '.jpg'
        img.save(file_name)
        self.generated_images_mux.unlock()

    def on_export_generated_video_clicked(self):
        self.generated_video_images_mux.lock()
        self.generated_images_mux.lock()
        file_name, _ = QFileDialog.getSaveFileName(
            self, caption='save to file', directory='./', filter='Video (*.gif)'
        )
        if file_name[-4:] in ['gif']:
            pass
        else:
            file_name = file_name + '.gif'
        self.video_preview_resolution_mux.lock()
        self.video_export_resolution_mux.lock()
        video_preview_resolution = self.ui.videoPreviewResolutionBox.currentText()
        video_export_resolution = self.ui.videoExportResolutionBox.currentText()
        self.video_export_resolution_mux.unlock()
        self.video_preview_resolution_mux.unlock()
        if video_preview_resolution != video_export_resolution:
            judge = len(self.generated_images)
            input_image = None
            if judge != 0:
                input_image = self.generated_images[0]
            if input_image is not None:
                input_image = pixmap2array(input_image)[..., :3]
                # generate
                self.video_text_prompts_mux.lock()
                self.video_export_resolution_mux.lock()
                text_prompts = self.video_text_prompts
                video_export_resolution = self.ui.videoExportResolutionBox.currentText().split('x')
                height = int(video_export_resolution[0].strip(' '))
                width = int(video_export_resolution[1].strip(' '))
                self.video_text_prompts_mux.unlock()
                self.video_export_resolution_mux.unlock()
                output_video = self.generate_pipeline.generate_video(
                    input_image,
                    text_prompts,
                    height = height,
                    width = width
                )
                save_videos_grid(output_video, file_name)
        else:
            video_frames = [pixmap2tensor(video_frame)[..., :3].permute([2, 0, 1]) for video_frame in self.generated_video_images]
            video_frames = torch.stack(video_frames, dim=1).unsqueeze(0)
            save_videos_grid_255(video_frames, file_name)
        self.generated_images_mux.unlock()
        self.generated_video_images_mux.unlock()

    def on_hide_controller_clicked(self):
        self.is_controller_visible = not self.is_controller_visible
        self.ui.controller.setVisible(self.is_controller_visible)
        if self.is_controller_visible:
            self.ui.hideControllerButton.setText('hide controller')
        else:
            self.ui.hideControllerButton.setText('show controller')
    
    def on_drag_mask_brush_radius_add_clicked(self):
        self.mask_brush_radius_mux.lock()
        if self.mask_brush_radius >= 100:
            pass
        else:
            self.mask_brush_radius += 5
        self.ui.dragMaskBrushRadiusText.setText(str(self.mask_brush_radius))
        self.mask_brush_radius_mux.unlock()
    
    def on_drag_mask_brush_radius_minus_clicked(self):
        self.mask_brush_radius_mux.lock()
        if self.mask_brush_radius <= 10:
            pass
        else:
            self.mask_brush_radius -= 5
        self.ui.dragMaskBrushRadiusText.setText(str(self.mask_brush_radius))
        self.mask_brush_radius_mux.unlock()
    
    def on_drag_motion_lambda_add_clicked(self):
        self.motion_lambda_mux.lock()
        if self.motion_lambda >= 100:
            pass
        else:
            self.motion_lambda += 5
        self.ui.dragMotionLambdaText.setText(str(self.motion_lambda))
        self.motion_lambda_mux.unlock()
    
    def on_drag_motion_lambda_minus_clicked(self):
        self.motion_lambda_mux.lock()
        if self.motion_lambda <= 10:
            pass
        else:
            self.motion_lambda -= 5
        self.ui.dragMotionLambdaText.setText(str(self.motion_lambda))
        self.motion_lambda_mux.unlock()
    
    def on_drag_flexible_area_clicked(self):
        self.mode_mux.lock()
        self.drag_markers_mux.lock()
        # self.generated_images_mux.lock()
        self.generator_params_mux.lock()
        if self.mode in DRAG_MODES:
            self.mode = DRAG_PAINT_FLEX_MASK_MODE
            if self.drag_markers[0]['mask'] is None:
                # if len(self.generated_images) > 0:
                #     H, W, C = pixmap2array(self.generated_images[0]).shape
                #     self.drag_markers[0]['mask'] = np.zeros((H, W), dtype=np.uint8)
                H, W = self.generator_params.image.height, self.generator_params.image.width
                self.drag_markers[0]['mask'] = np.zeros((H, W), dtype=np.uint8)
        else:
            pass
        # self.generated_images_mux.unlock()
        self.generator_params_mux.unlock()
        self.drag_markers_mux.unlock()
        self.mode_mux.unlock()
    
    def on_drag_fixed_area_clicked(self):
        self.mode_mux.lock()
        self.drag_markers_mux.lock()
        # self.generated_images_mux.lock()
        self.generator_params_mux.lock()
        if self.mode in DRAG_MODES:
            self.mode = DRAG_PAINT_FIX_MASK_MODE
            # if self.drag_markers[0]['mask'] is None and len(self.generated_images) > 0:
            #     H, W, C = pixmap2array(self.generated_images[0]).shape
            #     self.drag_markers[0]['mask'] = np.ones((H, W), dtype=np.uint8)
            H, W = self.generator_params.image.height, self.generator_params.image.width
            self.drag_markers[0]['mask'] = np.ones((H, W), dtype=np.uint8)
        else:
            pass
        # self.generated_images_mux.unlock()
        self.generator_params_mux.unlock()
        self.drag_markers_mux.unlock()
        self.mode_mux.unlock()
    
    def on_drag_reset_mask_clicked(self):
        self.drag_markers_mux.lock()
        self.mode_mux.lock()
        if self.drag_markers[0]['mask'] is None:
            pass
        else:
            if self.mode == DRAG_PAINT_FIX_MASK_MODE:
                self.drag_markers[0]['mask'] = np.ones_like(self.drag_markers[0]['mask'])
            elif self.mode == DRAG_PAINT_FLEX_MASK_MODE:
                self.drag_markers[0]['mask'] = np.zeros_like(self.drag_markers[0]['mask'])
        self.mode_mux.unlock()
        self.drag_markers_mux.unlock()
    
    def on_drag_show_mask_changed(self):
        self.is_show_mask_mux.lock()
        check_state = self.ui.dragShowMaskCheck.checkState()
        if check_state == Qt.Checked:
            self.is_show_mask = True
        else:
            self.is_show_mask = False
        self.is_show_mask_mux.unlock()
    
    def on_drag_add_point_click(self):
        self.mode_mux.lock()
        if self.mode in DRAG_MODES:
            self.mode = DRAG_ADD_POINT_MODE
        else:
            pass
        self.mode_mux.unlock()
    
    def on_drag_reset_point_click(self):
        self.drag_markers_mux.lock()
        self.drag_markers[0]['points'] = {}
        self.drag_markers_mux.unlock()
    
    def on_drag_start_click(self):
        self.mode_mux.lock()
        self.is_generate_drag_image_mux.lock()
        if self.mode in NORMAL_MODES:
            judge = False
        elif self.mode in DRAG_MODES:
            judge = not self.is_generate_drag_image
            self.is_generate_drag_image = True
        else:
            judge = False
        self.is_generate_drag_image_mux.unlock()
        self.mode_mux.unlock()
        if judge:
            self.drag_generation.start()
    
    def on_drag_stop_click(self):
        self.mode_mux.lock()
        self.is_generate_drag_image_mux.lock()
        if self.mode in NORMAL_MODES:
            pass
        elif self.mode in DRAG_MODES:
            self.is_generate_drag_image = False
        self.is_generate_drag_image_mux.unlock()
        self.mode_mux.unlock()
    
    def on_drag_mode_on_click(self):
        # TODO: support more state
        self.mode_mux.lock()
        if self.mode not in DRAG_MODES:
            self.mode = DRAG_ADD_POINT_MODE
            # do DragGAN initialization (mainly PTI inversion)
            self.generated_images_mux.lock()
            judge = len(self.generated_images)
            self.generated_images_mux.unlock()
            if judge == 0:
                self.mode = NORMAL_MODE  # no image for drag
            else:
                self.drag_preparing.start()
                self.drag_preparing.wait()
                # pass
        self.mode_mux.unlock()

    def on_drag_mode_off_click(self):
        # TODO: support more state
        self.mode_mux.lock()
        self.is_generate_drag_image_mux.lock()
        self.mode = NORMAL_MODE
        self.is_generate_drag_image = False
        self.is_generate_drag_image_mux.unlock()
        self.mode_mux.unlock()
    
    def on_confirm_text_prompt_click(self):
        self.image_text_prompts_mux.lock()
        self.video_text_prompts_mux.lock()
        self.is_image_text_prompt_uptodate_mux.lock()
        # pos
        image_pos_prompt = self.ui.imagePosPromptText.toPlainText()
        if len(image_pos_prompt) <= 0:
            image_pos_prompt = self.default_image_text_prompts[0]
        video_pos_prompt = self.ui.videoPosPromptText.toPlainText()
        if len(video_pos_prompt) <= 0:
            video_pos_prompt = self.default_video_text_prompts[0]
        
        # neg
        image_neg_prompt = self.ui.imageNegPromptText.toPlainText()
        if len(image_neg_prompt) <= 0:
            image_neg_prompt = self.default_image_text_prompts[1]
        video_neg_prompt = self.ui.videoNegPromptText.toPlainText()
        if len(video_neg_prompt) <= 0:
            video_neg_prompt = self.default_video_text_prompts[1]
        
        print('# image (pos ------ neg)')
        print(image_pos_prompt, ' ------ ', image_neg_prompt)
        print('# video (pos ------ neg)')
        print(video_pos_prompt, ' ------ ', video_neg_prompt)
        
        self.image_text_prompts = (image_pos_prompt, image_neg_prompt)
        self.video_text_prompts = (video_pos_prompt, video_neg_prompt)
        self.is_image_text_prompt_uptodate = False

        self.is_image_text_prompt_uptodate_mux.unlock()
        self.video_text_prompts_mux.unlock()
        self.image_text_prompts_mux.unlock()
    
    def on_clear_video_click(self):
        self.generated_video_images_mux.lock()
        self.generated_video_images = []
        self.generated_video_images_mux.unlock()
    
    def on_generate_video_click(self):
        self.is_generate_video_mux.lock()
        judge = self.is_generate_video
        self.is_generate_video = True
        self.is_generate_video_mux.unlock()
        if judge:
            pass
        else:
            # video generation will block the main thread
            video_generation = VideoGeneration(self)
            video_generation.start()
            video_generation.wait()
    
    def on_exit_click(self):
        self.is_exit = True  # exit signal
        time.sleep(0.05)
        # wait for sub threads to recieve exit signal
        self.capture.wait()
        if self.capture.isFinished():
            del self.capture
        self.image_viewer.wait()
        if self.image_viewer.isFinished():
            del self.image_viewer
        self.image_generation.wait()
        if self.image_generation.isFinished():
            del self.image_generation
        self.video_viewer.wait()
        if self.video_viewer.isFinished():
            del self.video_viewer
        self.generator_init.wait()
        if self.generator_init.isFinished():
            del self.generator_init
        sys.exit(0)


class MainWindowEventFilter(QObject):
    def __init__(self, win: MyWindow):
        super(MainWindowEventFilter, self).__init__()
        self.win = win

    def eventFilter(self, obj, event):
        win = self.win
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                win.draggable = True
                win.offset = event.globalPos() - win.pos()
        elif event.type() == QEvent.MouseMove:
            if win.draggable:
                win.move(event.globalPos() - win.offset)
        elif event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                win.draggable = False
        return False


class DragEditEventFilter(QObject):
    def __init__(self, win: MyWindow):
        super(DragEditEventFilter, self).__init__()
        self.win = win

    def eventFilter(self, obj, event):
        win = self.win
        if event.type() == QEvent.MouseButtonPress:
            # print(f'Mouse Pressed at ({event.pos().x()}, {event.pos().y()})')
            win.drag_markers_mux.lock()
            win.generator_params_mux.lock()
            win.mask_brush_radius_mux.lock()
            win.mode_mux.lock()
            H, W = win.generator_params.image.height, win.generator_params.image.width
            vH, vW = win.ui.imgOutput.height(), win.ui.imgOutput.width()
            pos_x = int(event.pos().x() * (W / vW))
            pos_y = int(event.pos().y() * (H / vH))
            if win.mode == DRAG_ADD_POINT_MODE:
                key_points = list(win.drag_markers[0]['points'].keys())
                key_points.sort(reverse=False)
                if len(key_points) == 0:  # no point pairs, add a new point pair
                    win.drag_markers[0]['points'][0] = {
                        'start_temp': [pos_x, pos_y],
                        'start': [pos_x, pos_y],
                        'target': None,
                    }
                else:
                    largest_id = key_points[-1]
                    if win.drag_markers[0]['points'][largest_id]['target'] is None:  # target is not set
                        win.drag_markers[0]['points'][largest_id]['target'] = [pos_x, pos_y]
                    else:  # target is set, add a new point pair
                        win.drag_markers[0]['points'][largest_id + 1] = {
                            'start_temp': [pos_x, pos_y],
                            'start': [pos_x, pos_y],
                            'target': None,
                        }
            elif win.mode in DRAG_PAINT_MASK_MODES:
                win.is_paint_mask = True
                if win.drag_markers[0]['mask'] is None:
                    if win.mode == DRAG_PAINT_FLEX_MASK_MODE:
                        win.drag_markers[0]['mask'] = np.zeros((H, W), dtype=np.uint8)
                    elif win.mode == DRAG_PAINT_FIX_MASK_MODE:
                        win.drag_markers[0]['mask'] = np.ones((H, W), dtype=np.uint8)
                if win.mode == DRAG_PAINT_FLEX_MASK_MODE:
                    win.drag_markers[0]['mask'] = draw_circle_on_mask(win.drag_markers[0]['mask'], pos_x, pos_y, win.mask_brush_radius, mode='add', inv=False)
                if win.mode == DRAG_PAINT_FIX_MASK_MODE:
                    win.drag_markers[0]['mask'] = draw_circle_on_mask(win.drag_markers[0]['mask'], pos_x, pos_y, win.mask_brush_radius, mode='mul', inv=True)
            
            win.mode_mux.unlock()
            win.mask_brush_radius_mux.unlock()
            win.generator_params_mux.unlock()
            win.drag_markers_mux.unlock()
        elif event.type() == QEvent.MouseMove:
            # print(f'Mouse Moved to ({event.pos().x()}, {event.pos().y()})')
            win.drag_markers_mux.lock()
            win.mode_mux.lock()
            win.mask_brush_radius_mux.lock()
            win.generator_params_mux.lock()
            H, W = win.generator_params.image.height, win.generator_params.image.width
            vH, vW = win.ui.imgOutput.height(), win.ui.imgOutput.width()
            pos_x = int(event.pos().x() * (W / vW))
            pos_y = int(event.pos().y() * (H / vH))
            if win.mode in DRAG_PAINT_MASK_MODES:
                if win.mode == DRAG_PAINT_FLEX_MASK_MODE:
                    win.drag_markers[0]['mask'] = draw_circle_on_mask(win.drag_markers[0]['mask'], pos_x, pos_y, win.mask_brush_radius, mode='add', inv=False)
                if win.mode == DRAG_PAINT_FIX_MASK_MODE:
                    win.drag_markers[0]['mask'] = draw_circle_on_mask(win.drag_markers[0]['mask'], pos_x, pos_y, win.mask_brush_radius, mode='mul', inv=True)
            win.generator_params_mux.unlock()
            win.mode_mux.unlock()
            win.mask_brush_radius_mux.unlock()
            win.drag_markers_mux.unlock()
        elif event.type() == QEvent.MouseButtonRelease:
            # print(f'Mouse Released at ({event.pos().x()}, {event.pos().y()})')
            win.mode_mux.lock()
            if win.mode in DRAG_PAINT_MASK_MODES:
                win.is_paint_mask = False
            win.mode_mux.unlock()
        else:
            return False
        return True


class Capture(QThread):
    """
        Capture the screen.
    """
    def __init__(self, win: MyWindow) -> None:
        super(Capture, self).__init__()
        self.win = win
    
    def run(self):
        win = self.win
        while not win.is_exit:
            # if in "drag" mode, Capture should be blocked
            win.mode_mux.lock()
            mode = win.mode
            win.mode_mux.unlock()
            if mode in NORMAL_MODES:
                pass
            elif mode in DRAG_MODES:
                time.sleep(0.05)
                continue
            elif mode in INIT_MODES:
                time.sleep(0.05)
                continue

            win_pos = win.pos()
            capture_rect = win.ui.imgCapture.geometry()
            viewer_rect = win.ui.viewer.geometry()
            screen= QApplication.primaryScreen()
            captured_image = screen.grabWindow(
                QApplication.desktop().winId(),
                win_pos.x() + capture_rect.x() + viewer_rect.x() + 3, win_pos.y() + capture_rect.y() + viewer_rect.y() + 3,
                capture_rect.width() - 6, capture_rect.height() - 6
            )
            captured_image = captured_image.toImage()  # RGB-32
            
            if captured_image.width() < 100 or captured_image.height() < 100:  # the are is to samll
                time.sleep(0.05)
                continue
            pixmap = QPixmap.fromImage(captured_image)

            # save captured image
            if CAPTURE_AUTO_SAVE:
                save_image = Image.fromarray(pixmap2array(pixmap)[..., :3])
                t = time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime())
                if not os.access(f'results/images/_tmp_cap/', os.F_OK):
                    os.makedirs(f'results/images/_tmp_cap/')
                save_image.save(f'results/images/_tmp_cap/{t}.jpg', 'JPEG')
                save_image.save(f'results/images/prompt.jpg', 'JPEG')

            win.captured_images_mux.lock()
            win.captured_images = [pixmap]
            win.captured_images_mux.unlock()
            time.sleep(0.05)


class ImageViewer(QThread):
    def __init__(self, win: MyWindow):
        super(ImageViewer, self).__init__()
        self.win = win
    
    def run(self):
        win = self.win
        while not win.is_exit:
            win.captured_images_mux.lock()
            if len(win.captured_images) == 0:
                pass
            else:
                piximg = win.captured_images[0]
                win.ui.imgOutput.setPixmap(piximg)
            win.captured_images_mux.unlock()
            time.sleep(0.05)


class GeneratedImageViewer(QThread):
    def __init__(self, win: MyWindow):
        super(GeneratedImageViewer, self).__init__()
        self.win = win
    
    def run(self):
        win = self.win
        while not win.is_exit:
            win.is_show_mask_mux.lock()
            win.drag_markers_mux.lock()
            win.mode_mux.lock()
            win.generated_images_mux.lock()
            if len(win.generated_images) == 0:
                pass
            else:
                # TODO: add mask in "render_view_image"
                piximg = win.generated_images[0]
                if win.mode in NORMAL_MODES:
                    pass
                elif win.mode in DRAG_MODES and len(win.drag_markers) > 0:
                    drag_markers = win.drag_markers[0]
                    piximg = render_view_image(piximg, drag_markers, show_mask=win.is_show_mask)
                win.ui.imgOutput.setPixmap(piximg)
            win.generated_images_mux.unlock()
            win.mode_mux.unlock()
            win.drag_markers_mux.unlock()
            win.is_show_mask_mux.unlock()
            time.sleep(0.05)


class GeneratedVideoViewer(QThread):
    def __init__(self, win: MyWindow):
        super(GeneratedVideoViewer, self).__init__()
        self.win = win
    
    def run(self):
        # TODO: add more controls to the video viewer
        win = self.win
        index = 0
        while not win.is_exit:
            win.generated_video_images_mux.lock()
            win.is_video_uptodate_mux.lock()
            length = len(win.generated_video_images)
            if length == 0:
                win.ui.videoOutput.setPixmap(QPixmap())
            elif win.is_video_uptodate:
                piximg = win.generated_video_images[index]
                win.ui.videoOutput.setPixmap(piximg)
                index = index + 1
                if index >= length:
                    index = 0
            else:
                index = 0
                win.is_video_uptodate = True
            win.is_video_uptodate_mux.unlock()
            win.generated_video_images_mux.unlock()
            time.sleep(0.1)


class GeneratorInit(QThread):
    def __init__(self, win: MyWindow) -> None:
        super(GeneratorInit, self).__init__()
        self.win = win
    
    def run(self):
        # initalize models
        self.win.generate_pipeline.init_model()
        print('[Model Initialized]')


class ImageGeneration(QThread):
    def __init__(self, win: MyWindow):
        super(ImageGeneration, self).__init__()
        self.win = win
    
    def run(self):
        win = self.win
        gen_count = 0
        while not win.is_exit:
            # if in "drag" mode, Image Generation should be blocked
            win.mode_mux.lock()
            mode = win.mode
            win.mode_mux.unlock()
            if mode in NORMAL_MODES:
                pass
            elif mode in DRAG_MODES:
                time.sleep(0.05)
                continue
            elif mode in INIT_MODES:
                time.sleep(0.05)
                continue

            win.captured_images_mux.lock()
            judge = len(win.captured_images)
            input_image = None
            if judge != 0:
                input_image = win.captured_images[0]
            win.captured_images_mux.unlock()
            if input_image is not None:
                win.is_image_text_prompt_uptodate_mux.lock()
                win.image_text_prompts_mux.lock()
                if win.is_image_text_prompt_uptodate:
                    text_prompts = None
                else:
                    text_prompts = win.image_text_prompts
                    win.is_image_text_prompt_uptodate = True
                win.image_text_prompts_mux.unlock()
                win.is_image_text_prompt_uptodate_mux.unlock()

                input_image = pixmap2tensor(input_image).permute([2, 0, 1])[:3, ...]
                # normalize
                input_image = input_image.to(torch.float32)
                input_image = (input_image / 255 - 0.5) * 2
                # generate
                output_image = win.generate_pipeline.generate_image(
                    input_image,
                    text_prompts,
                )[0]
                # denormalize
                output_image = output_image.permute([1, 2, 0])
                output_image = (output_image / 2 + 0.5).clamp(0, 1) * 255
                output_image = output_image.to(torch.uint8)

                # save generated image
                if gen_count == 0 and IGEN_AUTO_SAVE:
                    win.image_text_prompts_mux.lock()
                    save_image = Image.fromarray(output_image.cpu().numpy())
                    t = time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime())
                    if not os.access(f'results/images/_tmp/', os.F_OK):
                        os.makedirs(f'results/images/_tmp/')
                    save_image.save(f'results/images/_tmp/{win.image_text_prompts[0]}_{t}.jpg', 'JPEG')
                    win.image_text_prompts_mux.unlock()

                output_image = torch.cat([
                    output_image, 255 * torch.ones((output_image.shape[0], output_image.shape[1], 1), dtype=output_image.dtype, device=output_image.device)
                ], dim=2)
                
                output_image = QPixmap.fromImage(tensor2qimg(output_image))
                # output_image = win.captured_images[0]
                win.generated_images_mux.lock()
                win.generated_images = [output_image]
                win.generated_images_mux.unlock()
                gen_count += 1
                if gen_count == 10:
                    gen_count = 0
            time.sleep(0.05)


class VideoGeneration(QThread):
    def __init__(self, win: MyWindow):
        super(VideoGeneration, self).__init__()
        self.win = win
    
    def run(self):
        win = self.win
        win.generated_images_mux.lock()
        judge = len(win.generated_images)
        input_image = None
        if judge != 0:
            input_image = win.generated_images[0]
        win.generated_images_mux.unlock()
        if input_image is not None:
            input_image = pixmap2array(input_image)[..., :3]

            # generate
            win.video_text_prompts_mux.lock()
            win.video_preview_resolution_mux.lock()
            text_prompts = win.video_text_prompts
            video_preview_resolution = win.ui.videoPreviewResolutionBox.currentText().split('x')
            height = int(video_preview_resolution[0].strip(' '))
            width = int(video_preview_resolution[1].strip(' '))
            win.video_text_prompts_mux.unlock()
            win.video_preview_resolution_mux.unlock()
            output_video = win.generate_pipeline.generate_video(
                input_image,
                text_prompts,
                height = height,
                width = width
            )

            # save output video
            if VGEN_AUTO_SAVE:
                t = time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime())
                if not os.access(f"results/videos/_tmp", os.F_OK):
                    os.makedirs(f"results/videos/_tmp")
                save_videos_grid(output_video, f"results/videos/_tmp/{text_prompts[0]}_{t}.gif")

            output_video = output_video[0]
            output_video = output_video.clamp(0, 1) * 255
            output_video = output_video.to(torch.uint8)

            output_video = [
                tensor2qimg(
                    torch.cat([
                        output_video[:, i].permute([1, 2, 0]),
                        255 * torch.ones((output_video.shape[2], output_video.shape[3], 1), dtype=output_video.dtype, device = output_video.device)
                    ], dim=2
                    )
                )
                for i in range(output_video.shape[1])
            ]
            print('[video generation done]')
            # 3 T 512 512
            win.generated_video_images_mux.lock()
            win.is_video_uptodate_mux.lock()
            win.generated_video_images = [
                QPixmap.fromImage(output_video_image)
                for output_video_image in output_video
            ]
            win.is_video_uptodate = False
            win.is_video_uptodate_mux.unlock()
            win.generated_video_images_mux.unlock()
        
            win.is_generate_video_mux.lock()
            win.is_generate_video = False
            win.is_generate_video_mux.unlock()


class DragPreparing(QThread):
    def __init__(
        self, win: MyWindow
    ):
        super(DragPreparing, self).__init__()
        self.win = win
    
    def run(self):
        win = self.win

        win.generated_images_mux.lock()
        custom_image = win.generated_images[0]
        win.generated_images_mux.unlock()

        win.generator_params_mux.lock()
        win.draggan_checkpoints_mux.lock()
        custom_image = pixmap2array(custom_image)[..., :3]
        custom_image = Image.fromarray(custom_image)
        current_ckpt_name = win.ui.dragCheckpointBox.currentText()
        win.generate_pipeline.prepare_drag_model(
            custom_image,
            generator_params = win.generator_params,
            pretrained_weight = win.draggan_checkpoints[current_ckpt_name],
        )
        drag_base_image = np.array(win.generator_params.image)
        drag_base_image = np.concatenate(
            [
                drag_base_image,
                255 * np.ones((drag_base_image.shape[0], drag_base_image.shape[1], 1), dtype=np.uint8)
            ],
            axis=2
        )

        drag_base_image = array2qimg(drag_base_image)
        drag_base_image = QPixmap.fromImage(drag_base_image)

        win.generated_images_mux.lock()
        win.generated_images = [drag_base_image]
        win.generated_images_mux.unlock()

        win.generator_params_mux.unlock()
        win.draggan_checkpoints_mux.unlock()


class DragGeneration(QThread):
    def __init__(
        self, win: MyWindow
    ):
        super(DragGeneration, self).__init__()
        self.win = win
    
    def run(
        self,
    ):
        win = self.win
        go_on = False
        win.is_generate_drag_image_mux.lock()
        go_on = win.is_generate_drag_image
        win.is_generate_drag_image_mux.unlock()
        cur_step = 0
        win.ui.dragStepLabel.setText('Steps: {} / inf'.format(cur_step))

        win.drag_markers_mux.lock()
        win.generator_params_mux.lock()

        # print('drag generation start')
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()

        points = win.drag_markers[0]['points']
        if win.drag_markers[0]['mask'] is None:
            mask = np.ones((win.generator_params.image.height, win.generator_params.image.width), dtype=np.uint8)
        else:
            mask = win.drag_markers[0]['mask']
        win.generator_params_mux.unlock()
        win.drag_markers_mux.unlock()
        save_drag_image = None
        while go_on:  #  and cur_step < 300:  # and cur step condition
            win.is_generate_drag_image_mux.lock()
            go_on = win.is_generate_drag_image
            win.is_generate_drag_image_mux.unlock()
            # generate single image
            # TODO: this part is very noising!!! not fully understand the process now.
            # this part of code will of course crush

            # TODO: in the viewer, pay attention to the x-y order of points
            win.generator_params_mux.lock()
            win.motion_lambda_mux.lock()
            generated_image = win.generate_pipeline.drag_image(
                points,
                mask,
                motion_lambda = win.motion_lambda,
                generator_params = win.generator_params
            )
            win.motion_lambda_mux.unlock()
            win.generator_params_mux.unlock()
            generated_image = np.array(generated_image).astype(np.uint8)
            
            save_drag_image = generated_image
            if cur_step % 50 == 0:
                # save temporal image
                win.image_text_prompts_mux.lock()
                save_image = Image.fromarray(save_drag_image)
                t = time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime())
                if not os.access(f'results/images/_tmp/', os.F_OK):
                    os.makedirs(f'results/images/_tmp/')
                save_image.save(f'results/images/_tmp/{win.image_text_prompts[0]}_{t}_drag_{cur_step}.jpg', 'JPEG')
                win.image_text_prompts_mux.unlock()

            generated_image = np.concatenate(
                [
                    generated_image,
                    255 * np.ones((generated_image.shape[0], generated_image.shape[1], 1), dtype=np.uint8)
                ],
                axis = 2
            )
            generated_image = array2qimg(generated_image)
            win.generated_images_mux.lock()
            win.generated_images = [QPixmap.fromImage(generated_image)]
            win.generated_images_mux.unlock()

            # TODO: add "point" and "mask" to the view image
            win.drag_markers_mux.lock()
            win.drag_markers = [{'points': points, 'mask': mask}]
            win.drag_markers_mux.unlock()
            cur_step += 1
            win.ui.dragStepLabel.setText('Steps: {} / inf'.format(cur_step))
            if cur_step % 50 == 0:
                print('[{} / {}]'.format(cur_step, 'inf'))
        # torch.cuda.synchronize()
        # end_time = time.perf_counter()
        # print('drag generation end: {:.6f}'.format(end_time-start_time))
        
        # save dragged image
        if save_drag_image is not None:
            save_image = Image.fromarray(save_drag_image)
            t = time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime())
            if not os.access(f'results/images/_tmp/', os.F_OK):
                os.makedirs(f'results/images/_tmp/')
            save_image.save(f'results/images/_tmp/{t}_drag_final.jpg', 'JPEG')
        
        win.is_generate_drag_image_mux.lock()
        win.is_generate_drag_image = False
        win.is_generate_drag_image_mux.unlock()

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec())
from PIL import Image
from PIL import ImageDraw
import numpy as np


def draw_points_on_image(image,
                         points,
                         curr_point=None,
                         highlight_all=True,
                         radius_scale=0.01):
    overlay_rgba = Image.new("RGBA", image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay_rgba)
    for point_key, point in points.items():
        if ((curr_point is not None and curr_point == point_key)
                or highlight_all):
            p_color = (255, 0, 0)
            t_color = (0, 0, 255)

        else:
            p_color = (255, 0, 0, 35)
            t_color = (0, 0, 255, 35)

        rad_draw = int(image.size[0] * radius_scale)

        p_start = point.get("start_temp", point["start"])
        p_target = point["target"]

        if p_start is not None and p_target is not None:
            p_draw = int(p_start[0]), int(p_start[1])
            t_draw = int(p_target[0]), int(p_target[1])

            overlay_draw.line(
                (p_draw[0], p_draw[1], t_draw[0], t_draw[1]),
                fill=(255, 255, 0),
                width=2,
            )

        if p_start is not None:
            p_draw = int(p_start[0]), int(p_start[1])
            overlay_draw.ellipse(
                (
                    p_draw[0] - rad_draw,
                    p_draw[1] - rad_draw,
                    p_draw[0] + rad_draw,
                    p_draw[1] + rad_draw,
                ),
                fill=p_color,
            )

            if curr_point is not None and curr_point == point_key:
                # overlay_draw.text(p_draw, "p", font=font, align="center", fill=(0, 0, 0))
                overlay_draw.text(p_draw, "p", align="center", fill=(0, 0, 0))

        if p_target is not None:
            t_draw = int(p_target[0]), int(p_target[1])
            overlay_draw.ellipse(
                (
                    t_draw[0] - rad_draw,
                    t_draw[1] - rad_draw,
                    t_draw[0] + rad_draw,
                    t_draw[1] + rad_draw,
                ),
                fill=t_color,
            )

            if curr_point is not None and curr_point == point_key:
                # overlay_draw.text(t_draw, "t", font=font, align="center", fill=(0, 0, 0))
                overlay_draw.text(t_draw, "t", align="center", fill=(0, 0, 0))

    return Image.alpha_composite(image.convert("RGBA"),
                                 overlay_rgba).convert("RGB")


def draw_mask_on_image(image, mask):
    if mask is None:
        mask = np.ones((image.height, image.width), dtype=np.uint8)

    im_mask = np.uint8(mask * 255)
    im_mask_rgba = np.concatenate(
        (
            np.tile(im_mask[..., None], [1, 1, 3]),
            45 * np.ones(
                (im_mask.shape[0], im_mask.shape[1], 1), dtype=np.uint8),
        ),
        axis=-1,
    )
    im_mask_rgba = Image.fromarray(im_mask_rgba).convert("RGBA")

    return Image.alpha_composite(image.convert("RGBA"),
                                 im_mask_rgba).convert("RGB")


def draw_circle_on_mask(mask, x, y, radius, mode='add', inv=False):
    H, W = mask.shape
    J = np.arange(W, dtype=np.int32)
    I = np.arange(H, dtype=np.int32)
    I, J = np.meshgrid(I, J, indexing='ij')
    dis = (I - y)**2 + (J - x)**2
    if inv:
        new_mask = dis > radius**2
    else:
        new_mask = dis <= radius**2
    if mode == 'add':
        return (mask + new_mask).clip(0, 1)
    elif mode == 'mul':
        return mask * new_mask
    return (mask + new_mask).clip(0, 1)  # default add mode


def draw_circle_on_image(image, x, y, radius, color=(255, 0, 0)):
    H, W, C = image.shape
    J = np.arange(W, dtype=np.int32)
    I = np.arange(H, dtype=np.int32)
    I, J = np.meshgrid(I, J, indexing='ij')
    dis = (I - y)**2 + (J - x)**2
    mask = dis <= radius**2
    i_color = np.array(color, dtype=np.int32)
    i_color = np.expand_dims(i_color, axis=[0, 1])
    i_mask = mask.astype(np.int32)
    i_mask = np.expand_dims(i_mask, axis=[2])
    i_image = image.astype(np.int32)
    i_image = image + i_mask * i_color
    i_image = np.clip(i_image, 0, 255)
    return i_image.astype(np.uint8)
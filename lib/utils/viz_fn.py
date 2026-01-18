import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def get_color_bar(n, bar_type="winter"):
    color_bar = plt.get_cmap(bar_type)(np.arange(n) / n)[:, :3] * 256
    color_bar = color_bar.astype(np.uint8)
    return color_bar


def get_random_color(is_HEX=False, exclude=None, distance=50):
    color = np.random.randint(0, 256, [3])
    exclude = np.array(exclude) if exclude is not None else None
    if exclude is not None:
        while 1:
            flag = np.abs(color-exclude).max() < distance
            if flag:
                color = np.random.randint(0, 256, [3])
            else:
                break
    if is_HEX:
        color_hex = "#%02x%02x%02x" % (color[0], color[1], color[2])
        return color_hex
    else:
        return color


def depth_to_rgb(image, d_min=None, d_max=None, fake_color=False):
    # bina: (H, W)
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        image = image.copy()

    H, W = image.shape[0], image.shape[1]
    d_min = image.min() if d_min is None else d_min
    d_max = image.max() if d_max is None else d_max

    rgb = np.zeros([H, W, 3], dtype=np.uint8)
    image = (image - d_min) / (d_max - d_min + 1e-5) * 255
    image = image.astype(np.uint8)

    if fake_color:
        color_bar = get_color_bar(256)
        rgb = color_bar[image]
    else:
        image = image[..., None]
        rgb = rgb + image

    return rgb


def draw_pts_on_image(image, pts, color=None, radius=1, thickness=2):
    # image: (H, W) or (H, W, 3)
    # pts: (N, 2) or (N, 3)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        image = image.copy()

    if len(image.shape) == 2:
        image = depth_to_rgb(image)

    color_bar = get_color_bar(pts.shape[0])
    for i, pt in enumerate(pts):
        pt = int(pt[0]), int(pt[1])
        if color is None:
            color_ = color_bar[i].tolist()
        else:
            color_ = color
        cv2.circle(image, pt, radius=radius, color=color_, thickness=thickness)

    return image


def draw_bbox_on_image(image, bbox, color=None, thickness=2):
    # image: (H, W) or (H, W, 3)
    # bbox: (..., 4)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        image = image.copy()

    if len(image.shape) == 2:
        image = depth_to_rgb(image)

    is_multibbox = len(bbox.shape) == 2

    if is_multibbox:
        for i, bb in enumerate(bbox):
            bb = bb.astype(np.int32)
            if color is None:
                color_ = get_random_color()
            else:
                color_ = color
            cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), color_, thickness=thickness)
    else:
        bb = bbox.astype(np.int32)
        if color is None:
            color_ = get_random_color()
        else:
            color_ = color
        cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), color_, thickness=thickness)
    return image


# from 2020_CVPR_HigherHRNet
def make_heatmaps(image, heatmaps):
    heatmaps = np.clip(heatmaps * 255, 0, 255).astype(np.uint8)

    num_joints, height, width = heatmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap*0.7 + image_resized*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid
    

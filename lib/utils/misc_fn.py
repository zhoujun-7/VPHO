import numpy as np
import torch
import contextlib
import warnings
import cv2
import os
from natsort import natsorted

def dep_to_3channel_inv(image):
    # image: (H, W, 3)
    out_image = np.zeros([image.shape[0], image.shape[1]], dtype=np.int64)
    out_image = out_image + image[..., 0] * 256**2 + image[..., 1] * 256 + image[..., 2]
    return out_image


def dep_to_3channel(image):
    # image: (H, W)
    image = image.copy().astype(np.int64)
    out_image = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    out_image[..., 0] = image // 256**2
    out_image[..., 1] = image // 256
    out_image[..., 2] = image % 256
    return out_image


def get_bbox3d_from_verts(verts):
    # verts: (..., N, 3)
    # return: (..., 2, 3)
    if isinstance(verts, np.ndarray):
        min_xyz = np.min(verts, axis=-2)
        max_xyz = np.max(verts, axis=-2)
        bbox3d = np.stack([min_xyz, max_xyz], axis=-2)
    elif isinstance(verts, torch.Tensor):
        min_xyz = torch.min(verts, dim=-2)[0]
        max_xyz = torch.max(verts, dim=-2)[0]
        bbox3d = torch.stack([min_xyz, max_xyz], dim=-2)
    else:
        raise NotImplementedError
    return bbox3d # (..., 2, 3)


def get_obj_kpt27_from_bbox3d(bbox3d):
    # bbox3d: (..., 2, 3)
    # return: (..., 27, 3)
    min_xyz = bbox3d[..., 0, :]
    max_xyz = bbox3d[..., 1, :]
    if isinstance(bbox3d, np.ndarray):
        kpt_ls = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    w = np.array([i, j, k]) / 2
                    kpt = min_xyz + w * (max_xyz - min_xyz)
                    kpt_ls.append(kpt)
        kpt27 = np.stack(kpt_ls, axis=-2)
    elif isinstance(bbox3d, torch.Tensor):
        kpt_ls = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    w = torch.tensor([i, j, k], device=bbox3d.device) / 2
                    kpt = min_xyz + w * (max_xyz - min_xyz)
                    kpt_ls.append(kpt)
        kpt27 = torch.stack(kpt_ls, dim=-2)
    else:
        raise NotImplementedError
    return kpt27 # (..., 27, 3)


# from 2023_CVPR_HFL
def fuse_bbox(bbox_1, bbox_2, img_shape, scale_factor=1.):
    bbox = np.concatenate((bbox_1.reshape(2, 2), bbox_2.reshape(2, 2)), axis=0)
    min_x, min_y = bbox.min(0)
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = bbox.max(0)
    max_x, max_y = min(max_x, img_shape[0]), min(max_y, img_shape[1])
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    center = np.asarray([c_x, c_y])
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    max_delta = max(delta_x, delta_y)
    scale = max_delta * scale_factor
    return center, scale


def expand_bbox2d(bbox_x1y1x2y2, scale_factor=1):
    # bbox_x1y1x2y2: (4, )
    bbox_cxcywh = bbox_x1y1x2y2.copy()
    bbox_cxcywh[:2] = (bbox_x1y1x2y2[:2] + bbox_x1y1x2y2[2:]) / 2
    bbox_cxcywh[2:] = bbox_x1y1x2y2[2:] - bbox_x1y1x2y2[:2]
    bbox_cxcywh[2:] *= scale_factor
    bbox_x1y1x2y2 = np.zeros(4)
    bbox_x1y1x2y2[:2] = bbox_cxcywh[:2] - bbox_cxcywh[2:] / 2
    bbox_x1y1x2y2[2:] = bbox_cxcywh[:2] + bbox_cxcywh[2:] / 2
    return bbox_x1y1x2y2



def pt2d_to_bbox2d(pts2d, mode="x1y1x2y2"):
    """ pts2d: (..., 21, 2)
        mode: "x1y1x2y2" or "x1y1wh" or "cxcywh"
    """
    if isinstance(pts2d, np.ndarray):
        x_min = np.min(pts2d[..., 0], axis=-1)
        x_max = np.max(pts2d[..., 0], axis=-1)
        y_min = np.min(pts2d[..., 1], axis=-1)
        y_max = np.max(pts2d[..., 1], axis=-1)
        if mode == "x1y1x2y2":
            return np.stack([x_min, y_min, x_max, y_max], axis=-1)
        elif mode == "x1y1wh":
            w = x_max - x_min
            h = y_max - y_min
            return np.stack([x_min, y_min, w, h], axis=-1)
        elif mode == "cxcywh":
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min
            return np.stack([cx, cy, w, h], axis=-1)
        
    elif isinstance(pts2d, torch.Tensor):
        x_min = torch.min(pts2d[..., 0], dim=-1)[0]
        x_max = torch.max(pts2d[..., 0], dim=-1)[0]
        y_min = torch.min(pts2d[..., 1], dim=-1)[0]
        y_max = torch.max(pts2d[..., 1], dim=-1)[0]
        if mode == "x1y1x2y2":
            return torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        elif mode == "x1y1wh":
            w = x_max - x_min
            h = y_max - y_min
            return torch.stack([x_min, y_min, w, h], dim=-1)
        elif mode == "cxcywh":
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min
            return torch.stack([cx, cy, w, h], dim=-1)
        
    elif isinstance(pts2d, list):
        jt2d_np = np.array(pts2d)
        bbox2d_np = pt2d_to_bbox2d(jt2d_np, mode=mode)
        return bbox2d_np.tolist()
    else:
        raise NotImplementedError
    

def protect_bbox2d(bbox2d, image, mode='xyxy'):
    """ bbox2d: (4, )
        image: (H, W, ...)
    """
    assert mode == 'xyxy'
    bbox2d[0] = max(0, bbox2d[0])
    bbox2d[1] = max(0, bbox2d[1])
    bbox2d[2] = min(image.shape[1], bbox2d[2])
    bbox2d[3] = min(image.shape[0], bbox2d[3])
    return bbox2d

def check_bbox2d(bbox2d, image, mode='xyxy'):
    """ bbox2d: (4, )
        image: (H, W, ...)
    """
    flag = True
    flag = flag and bbox2d[0] >= 0
    flag = flag and bbox2d[1] >= 0
    flag = flag and bbox2d[2] <= image.shape[1]
    flag = flag and bbox2d[3] <= image.shape[0]
    flag = flag and bbox2d[0] < bbox2d[2]
    flag = flag and bbox2d[1] < bbox2d[3]
    return flag

def get_unite_bbox2d(bbox2d1, bbox2d2, mode='xyxy'):
    """
        Args:
            bbox2d1: (..., 4)
            bbox2d2: (..., 4)
            mode: 'xyxy'
        Reture:
            bbox2d: (..., 4)
    """
    assert mode == 'xyxy'
    if isinstance(bbox2d1, torch.Tensor) and isinstance(bbox2d2, torch.Tensor):
        bbox2d = torch.stack([torch.min(bbox2d1[..., 0], bbox2d2[..., 0]),
                              torch.min(bbox2d1[..., 1], bbox2d2[..., 1]),
                              torch.max(bbox2d1[..., 2], bbox2d2[..., 2]),
                              torch.max(bbox2d1[..., 3], bbox2d2[..., 3])], dim=-1)
    elif isinstance(bbox2d1, np.ndarray) and isinstance(bbox2d2, np.ndarray):
        bbox2d = np.stack([np.minimum(bbox2d1[..., 0], bbox2d2[..., 0]),
                          np.minimum(bbox2d1[..., 1], bbox2d2[..., 1]),
                          np.maximum(bbox2d1[..., 2], bbox2d2[..., 2]),
                          np.maximum(bbox2d1[..., 3], bbox2d2[..., 3])], axis=-1)
    else:
        raise NotImplementedError
    return bbox2d


def get_inter_bbox2d(bbox2d1, bbox2d2, mode='xyxy'):
    """
        Args:
            bbox2d1: (..., 4)
            bbox2d2: (..., 4)
            mode: 'xyxy'
        Reture:
            bbox2d: (..., 4)
    """
    assert mode == 'xyxy'
    if isinstance(bbox2d1, torch.Tensor) and isinstance(bbox2d2, torch.Tensor):
        bbox2d = torch.stack([torch.max(bbox2d1[..., 0], bbox2d2[..., 0]),
                              torch.max(bbox2d1[..., 1], bbox2d2[..., 1]),
                              torch.min(bbox2d1[..., 2], bbox2d2[..., 2]),
                              torch.min(bbox2d1[..., 3], bbox2d2[..., 3])], dim=-1)
    elif isinstance(bbox2d1, np.ndarray) and isinstance(bbox2d2, np.ndarray):
        bbox2d = np.stack([np.maximum(bbox2d1[..., 0], bbox2d2[..., 0]),
                          np.maximum(bbox2d1[..., 1], bbox2d2[..., 1]),
                          np.minimum(bbox2d1[..., 2], bbox2d2[..., 2]),
                          np.minimum(bbox2d1[..., 3], bbox2d2[..., 3])], axis=-1)
    else:
        raise NotImplementedError
    return bbox2d


def get_rectanglular_bbox2d(bbox2d, mode='xyxy'):
    """
        Args:
            bbox2d: (..., 4)
            mode: 'xyxy'
        Reture:
            bbox2d: (..., 4)
    """
    assert mode == 'xyxy'
    if isinstance(bbox2d, torch.Tensor):
        bbox2d = bbox2d.clone()
        ct = (bbox2d[..., :2] + bbox2d[..., 2:]) / 2
        wh = bbox2d[..., 2:] - bbox2d[..., :2]
        max_wh = torch.max(wh, dim=-1)[0]
        bbox2d[..., :2] = ct - max_wh / 2
        bbox2d[..., 2:] = ct + max_wh / 2
    elif isinstance(bbox2d, np.ndarray):
        bbox2d = bbox2d.copy()
        ct = (bbox2d[..., :2] + bbox2d[..., 2:]) / 2
        wh = bbox2d[..., 2:] - bbox2d[..., :2]
        max_wh = np.max(wh, axis=-1)
        bbox2d[..., :2] = ct - max_wh / 2
        bbox2d[..., 2:] = ct + max_wh / 2
    else:
        raise NotImplementedError
    return bbox2d, max_wh


def to_device(sample, device):
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.float64:
                sample[k] = v.float().to(device)
            else:
                sample[k] = v.to(device)
    return sample


def to_numpy(sample):
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            sample[k] = v.detach().cpu().numpy()
        else:
            sample[k] = np.array(v)
    return sample

def to_tensor(sample):
    if isinstance(sample, dict):
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = torch.from_numpy(v)
            elif isinstance(v, dict):
                sample[k] = to_tensor(v)
    elif isinstance(sample, list):
        for i, v in enumerate(sample):
            if isinstance(v, np.ndarray):
                sample[i] = torch.from_numpy(v)
            elif isinstance(v, dict):
                sample[i] = to_tensor(v)
    return sample


#* checked, modified from HigherHRNet
class HeatmapGenerator():
    def __init__(self, output_res, sigma=-1):
        self.output_res = output_res
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        num_joints = joints.shape[0]
        hms = np.zeros((num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma

        for idx, pt in enumerate(joints):
            if allow_subpixel := False:
                x, y = pt[0], pt[1]
            else:
                x, y = int(pt[0]), int(pt[1])
            if x < 0 or y < 0 or \
                x >= self.output_res or y >= self.output_res:
                continue

            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.output_res)
            aa, bb = max(0, ul[1]), min(br[1], self.output_res)
            hms[idx, aa:bb, cc:dd] = np.maximum(
                hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms
    
    def get_heatmap(self, pt2d, bbox, is_right=True):
        max_wh = max(bbox[2:] - bbox[:2])
        pt2d_hm = pt2d - bbox[:2]
        pt2d_hm = pt2d_hm / (max_wh) * (self.output_res - 1)
        pt2d_hm[:, 0] = pt2d_hm[:, 0] + 1 if not is_right else pt2d_hm[:, 0]
        heatmap = self(pt2d_hm)
        return heatmap
    
# generete heatmap for Arbitrary Rectangle
class AdaptiveHeatmapGenerator():
    def __init__(self, num_joints, sigma=-1, hm_size=64):
        self.num_joints = num_joints
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.hm_size = hm_size

    def get_heatmap(self, joints, tight_bbx_res):
        num_joints = joints.shape[0]
        hms = np.zeros((num_joints, tight_bbx_res[1],tight_bbx_res[0]),
                       dtype=np.float32)
        sigma = self.sigma

        for idx, pt in enumerate(joints):
            if allow_subpixel := False:
                x, y = pt[0], pt[1]
            else:
                x, y = int(pt[0]), int(pt[1])
            if x < 0 or y < 0 or \
                x >= tight_bbx_res[0] or y >= tight_bbx_res[1]:
                continue

            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            c, d = max(0, -ul[0]), min(br[0], tight_bbx_res[0]) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], tight_bbx_res[1]) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], tight_bbx_res[0])
            aa, bb = max(0, ul[1]), min(br[1], tight_bbx_res[1])

            hms[idx, aa:bb, cc:dd] = np.maximum(
                hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

    def __call__(self, jt2d, tight_bbx_res):
        w, h = tight_bbx_res[2:] - tight_bbx_res[:2]
        max_l = max(w, h)
        res = (int(self.hm_size*w/max_l), int(self.hm_size*h/max_l))
        
        jt2d_hm = jt2d.copy()
        jt2d_hm[:, 0] = (jt2d_hm[:, 0] - tight_bbx_res[0]) * res[0] / w
        jt2d_hm[:, 1] = (jt2d_hm[:, 1] - tight_bbx_res[1]) * res[1] / h
        heatmap = self.get_heatmap(jt2d_hm, res)
        heatmap = cv2.resize(heatmap.transpose(1, 2, 0), 
                             (self.hm_size, self.hm_size), 
                             interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        heatmap[heatmap < self.g.min()] = 0
        return heatmap


@contextlib.contextmanager
def gpu_running_timer():
    """ Count the gpu running time.

    Usage:
        >>> with gpu_running_timer():
                ...(gpu operation)

    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    yield
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('Elapsed time: {:.3f} ms'.format(elapsed_time_ms))


def deprecated(reason, substitute):
    def decorator(func):
        def new_func(*args, **kwargs):
            warnings.warn("{} has be deprecated. {} Please use {} instead.".format(func.__name__, reason, substitute), 
                          category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return new_func
    return decorator


def bx2d2_to_bx2d4(bx2d2):
    """ xyxy mode
        bx2d2: (..., 2, 2)
        bx2d4: (..., 4, 2)
    """
    idx1 = [[0, 0], [1, 0], [1, 1], [0, 1]]
    idx2 = [[0, 1], [0, 1], [0, 1], [0, 1]]
    bx2d4 = bx2d2[..., idx1, idx2]
    return bx2d4

def search_files(root_dir, file_ext_ls=[]):
    f_ls = natsorted(os.listdir(root_dir))
    f_ls = [os.path.join(root_dir, f) for f in f_ls]

    all_f_ls = []
    for f in f_ls:
        if os.path.isdir(f):
            ff_ls = search_files(f, file_ext_ls)
            all_f_ls.extend(ff_ls)
        else:
            n, ext = os.path.splitext(f)
            if isinstance(file_ext_ls, list):
                if ext in file_ext_ls:
                    all_f_ls.append(f)
            elif file_ext_ls == "any":
                all_f_ls.append(f)
    return all_f_ls

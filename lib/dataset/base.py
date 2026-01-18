import os
import cv2
import json
import time
import torch
import pickle
import trimesh
import fpsample
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from natsort import natsorted
from torch.utils.data import Dataset
from collections import defaultdict
from manopth.manolayer import ManoLayer
from scipy.spatial.distance import cdist
from timm.data.random_erasing import RandomErasing
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

from lib.utils.hand_fn import fill_finger_gaps_in_mano, JOINT_VERTS_IDX, FINGER_VERTS_IDX, FINGER_IDX
from lib.utils.render_fn import Pytorch3DRenderer
from lib.utils.misc_fn import (
    get_bbox3d_from_verts, 
    get_obj_kpt27_from_bbox3d, 
    expand_bbox2d, 
    fuse_bbox, 
    dep_to_3channel, 
    dep_to_3channel_inv, 
    pt2d_to_bbox2d,
    HeatmapGenerator,
    AdaptiveHeatmapGenerator,
    get_rectanglular_bbox2d,
)
from lib.utils.physics_fn import detect_thin_shell, detect_hand_and_object_contact
from lib.utils.viz_fn import get_random_color, depth_to_rgb
from lib.configs.args import Config


YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}
YCB_ID = {v: k for k, v in YCB_CLASSES.items()}

THIN_SHELL_CLASSES = [
    '019_pitcher_base',
    '024_bowl',
    '025_mug',
    '037_scissors',
    '051_large_clamp',
    '052_extra_large_clamp',
]

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

IMG_MEAN = IMAGENET_DEFAULT_MEAN
IMG_STD = IMAGENET_DEFAULT_STD


mano_layer_r = ManoLayer(flat_hand_mean=True, side="right", mano_root="asset/mano_v1_2/models", use_pca=False)
mano_layer_l = ManoLayer(flat_hand_mean=True, side="left", mano_root="asset/mano_v1_2/models", use_pca=False)

# vert2joint = mano_layer_r.th_J_regressor.numpy()
# tip_mat = np.zeros([5, 778])
# tip_mat[[0,1,2,3,4], [745, 320, 444, 556, 673]] = 1
# vert2joint = np.concatenate([vert2joint, tip_mat], 0)
# vert2joint = vert2joint[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
# save_dt = {"vert2joint": vert2joint}
# with open("asset/ours/vert2joint.pkl", "wb") as f:
#     pickle.dump(save_dt, f)

def normalize_rgb(rgb):
    rgb = rgb.astype(np.float32)
    mean = np.array(IMG_MEAN, dtype=np.float32)
    std = np.array(IMG_STD, dtype=np.float32)
    rgb = (rgb / 255.0 - mean) / std
    return rgb


def normalize_depth(depth):
    depth = depth / 10.0  # in decimeter
    return depth


def normalize_contact(joint_map, weight_map):
    # joint_map: (H, W)
    # weight_map: (H, W)
    contact_map = np.zeros_like([7, joint_map.shape[0], joint_map.shape[1]], dtype=np.bool_)
    for i in range(6):
        contact_map[i] = (joint_map == i)
    contact_map[6] = weight_map
    return contact_map


def inverse_normalize_rgb(rgb):
    rgb = rgb * np.array(IMG_STD, dtype=np.float32) + np.array(IMG_MEAN, dtype=np.float32)
    rgb = np.clip(rgb, 0, 1) * 255
    return rgb.astype(np.uint8)


def np_to_tensor(np_array, is_img=False):
    if is_img:
        np_array = np_array.transpose(2, 0, 1)
        return torch.from_numpy(np_array).float()
    else:
        return torch.from_numpy(np_array).float()
    
    
def tensor_to_np(tensor, is_img=False):
    if is_img:
        tensor = tensor.cpu().detach().numpy()
        tensor = tensor.transpose(1, 2, 0)
        return tensor
    else:
        return tensor.cpu().detach().numpy()
    

def get_hand_vert(mano_pose_aa_flat, mano_shape, rot, transl, is_right=False):
    """ 
        Args:
            mano_pose_aa_flat: (batch, 45)
            mano_shape: (batch, 10)
            rot: (batch, 3)
            transl: (batch, 3)
        Return:
            hand_verts: (batch,  V, 3) np.float32
            jt3d:       (batch, 21, 3) np.float32
    """
    pose_coeffs = np.concatenate([rot, mano_pose_aa_flat], axis=-1)
    th_pose_coeffs = torch.from_numpy(pose_coeffs).float()[None]
    th_betas=torch.from_numpy(mano_shape).float()[None]
    th_trans=torch.from_numpy(transl).float()[None]

    if is_right:
        hand_verts, jt3d = mano_layer_r(th_pose_coeffs, th_betas, th_trans)
    else:
        hand_verts, jt3d = mano_layer_l(th_pose_coeffs, th_betas, th_trans)
    hand_verts = hand_verts.squeeze().detach().cpu().numpy() / 1000.0
    jt3d = jt3d.squeeze().detach().cpu().numpy() / 1000.0
    return hand_verts, jt3d


def bx3d2_to_bx3d8(bx3d2):
    """
        Args: 
            bx3d2: (2, 3)
        Return:
            bx3d8: (8, 3)
    """
    x1, y1, z1 = bx3d2[0]
    x2, y2, z2 = bx3d2[1]
    bx3d8 = np.array([
        [x1, y1, z1],
        [x1, y1, z2],
        [x1, y2, z1],
        [x1, y2, z2],
        [x2, y1, z1],
        [x2, y1, z2],
        [x2, y2, z1],
        [x2, y2, z2],
    ])
    return bx3d8


#* from 2023_CVPR_HFL
def get_diameter(vertex):
    vp = vertex[:]
    x = vp[:, 0].reshape((1, -1))
    y = vp[:, 1].reshape((1, -1))
    z = vp[:, 2].reshape((1, -1))
    x_max, x_min, y_max, y_min, z_max, z_min = np.max(x), np.min(x), np.max(y), np.min(y), np.max(z), np.min(z)
    diameter_x = abs(x_max - x_min)
    diameter_y = abs(y_max - y_min)
    diameter_z = abs(z_max - z_min)
    diameters = np.sqrt(diameter_x ** 2 + diameter_y ** 2 + diameter_z ** 2)
    return diameters


def get_object_mesh_dt(model_dir):
    cache_path = os.path.join("asset/ours", "object_mesh_info.pkl")
    if os.path.exists(cache_path):
    # if False:
        with open(cache_path, "rb") as f:
            obj_mesh_dt = pickle.load(f)
    else:
        print("Building object mesh info...")
        t1 = time.time()
        with open("asset/ours/object_shift_to_axial_symmetry.json", "r") as f:
            Rt_shift_to_ax_sym = json.load(f)
        with open("asset/ours/object_center_of_mass.json", "r") as f:
            obj_center_of_mass = json.load(f)
        obj_name_ls = natsorted(os.listdir(model_dir))
        obj_path_ls = [os.path.join(model_dir, obj_name, "textured_simple.obj") for obj_name in obj_name_ls]
        obj_mesh_dt = defaultdict(dict)
        for obj_name, obj_path in zip(obj_name_ls, obj_path_ls):
            obj_mesh_dt[obj_name]["mesh"] = trimesh.load(obj_path)
            obj_mesh_dt[obj_name]["verts"] = obj_mesh_dt[obj_name]["mesh"].vertices.copy()
            obj_mesh_dt[obj_name]["faces"] = obj_mesh_dt[obj_name]["mesh"].faces.copy()
            normals= obj_mesh_dt[obj_name]["mesh"].vertex_normals
            obj_mesh_dt[obj_name]["normals"] = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-20)
            obj_mesh_dt[obj_name]["shift"] = np.array(Rt_shift_to_ax_sym[obj_name])
            obj_mesh_dt[obj_name]["CoM"] = np.array(obj_center_of_mass[obj_name])
            verts_sampled_index = fpsample.bucket_fps_kdline_sampling(obj_mesh_dt[obj_name]["verts"], 2048, h=3, start_idx=0)
            verts_sampled = obj_mesh_dt[obj_name]["verts"][verts_sampled_index]
            normals_sampled = obj_mesh_dt[obj_name]["normals"][verts_sampled_index]
            obj_mesh_dt[obj_name]["verts_sampled"] = verts_sampled
            obj_mesh_dt[obj_name]["normals_sampled"] = normals_sampled
            verts_sampled_axsym = verts_sampled @ obj_mesh_dt[obj_name]["shift"][:3, :3].T + obj_mesh_dt[obj_name]["shift"][:3, 3]
            obj_mesh_dt[obj_name]["verts_sampled_axsym"] = verts_sampled_axsym
            bbox3d_axsym = get_bbox3d_from_verts(verts_sampled_axsym)
            bbox3d8_axsym = bx3d2_to_bx3d8(bbox3d_axsym)
            bbox3d8 = (bbox3d8_axsym - obj_mesh_dt[obj_name]["shift"][:3, 3]) @ obj_mesh_dt[obj_name]["shift"][:3, :3]
            obj_mesh_dt[obj_name]["bbox3d"] = bbox3d8
            obj_mesh_dt[obj_name]["bbox3d_axsym"] = bbox3d8_axsym
            obj_kpt27_axsym = get_obj_kpt27_from_bbox3d(bbox3d_axsym)
            obj_mesh_dt[obj_name]["kpt3d_axsym"] = obj_kpt27_axsym
            obj_kpt27 = (obj_kpt27_axsym - obj_mesh_dt[obj_name]["shift"][:3, 3]) @ obj_mesh_dt[obj_name]["shift"][:3, :3]
            obj_mesh_dt[obj_name]["kpt3d"] = obj_kpt27
            obj_mesh_dt[obj_name]["diameter"] = get_diameter(obj_mesh_dt[obj_name]["mesh"].vertices)  # align with 2023-CVPR-HFL
            # if obj_name in THIN_SHELL_CLASSES:
            #     thin_shell_pair = detect_thin_shell(obj_mesh_dt[obj_name]["verts"], 
            #                                         obj_mesh_dt[obj_name]["normals"], 
            #                                         shell_thickness_thresh=[0.002, 0.01], 
            #                                         normal_angle_thresh=150)
            #     obj_mesh_dt[obj_name]["thin_shell"] = thin_shell_pair
            # else:
            #     obj_mesh_dt[obj_name]["thin_shell"] = None
        with open(cache_path, "wb") as f:
            pickle.dump(obj_mesh_dt, f)
        t2 = time.time()
        print(f"Building object mesh info done! Time: {t2 - t1:.2f}s")
    return obj_mesh_dt
YCB_MESHES = get_object_mesh_dt("/root/Workspace/HOI/data/DexYCB/models")


#* from 2023_CVPR_HFL
def check_bbox2d(bbox_xyxy, img_width, img_height):
    """ 
        Args:
            bbox_xyxy: (4, )
            img_width: int
            img_height: int
        Return:
            is_ok: bool
    """
    is_ok = True
    x, y, w, h = bbox_xyxy
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox_xyxy = np.array([x1, y1, x2-x1, y2-y1])
    else:
        is_ok = False
    return is_ok


#* from 2023_CVPR_HFL    checked
def filter_DexYCB_by_HFL(sample):
    """ has both right and left hands
        Training samples: 401507
        Testing samples: 78005
        Return:
            is_ok: bool
    """
    jt2d = np.array(sample["joint_2d"],dtype=np.float32).squeeze()
    hand_bbox = pt2d_to_bbox2d(jt2d, mode="cxcywh")
    expand_ratio = 1.5
    hand_bbox[..., 2:] *= expand_ratio
    hand_bbox[..., :2] -= hand_bbox[..., 2:] / 2 # x1y1wh
    is_ok = check_bbox2d(hand_bbox, 640, 480)
    return is_ok


#* from 2022_CVPR_ArtiBoost    checked
def filter_DexYCB_by_ArtiBoost(sample, use_left_hand=False, filter_no_contact=True, filter_invisible_hand=True, filter_thresh=50):
    """ has only right hand
        Training samples: 168280
        Testing samples: 32968
        Return:
            is_ok: bool
    """
    is_ok = True
    if not use_left_hand and sample["mano_side"] == "left":
        is_ok = False

    jt2d = np.array(sample["joint_2d"],dtype=np.float32).squeeze()
    if filter_invisible_hand and np.all(jt2d == -1.0):
        is_ok = False
    
    jt3d = np.array(sample["joint_3d"],dtype=np.float32).squeeze()
    object_6D = np.array(sample["pose_y"][sample['ycb_grasp_ind']],dtype=np.float32) # (3, 4)
    object_id = sample["ycb_ids"][sample['ycb_grasp_ind']]
    object_name = YCB_CLASSES[object_id]
    object_verts = YCB_MESHES[object_name]["verts"]
    object_verts_transf = object_verts @ object_6D[:3, :3].T + object_6D[:3, 3]
    ho_distance = cdist(object_verts_transf, jt3d)
    if filter_no_contact and (ho_distance.min() * 1000.0 > filter_thresh):
        is_ok = False
    return is_ok


class Filter_DexYCB_by_Steady_Grasping:
    """ has both right and left hands
        Training samples: 237256
        Testing samples: 45624
        Return:
            is_ok: bool
    """
    def __init__(self):
        with open("asset/ours/DexYCB/is_off_desk_5cm.pkl", "rb") as f:
            self.is_off_desk_dt = pickle.load(f)

    def __call__(self, sample):
        color_file = sample["color_file"]
        dir_names = color_file.split("/")
        sequence_key = dir_names[0] + "/" + dir_names[1]
        sequ_id = int(dir_names[-1].split('.')[0].split('_')[-1])
        is_ok = is_off_desk = self.is_off_desk_dt[sequence_key][sequ_id]
        return is_ok


class ImageAugmentor:
    _cfg = {
        'clahe_prob': 0.5,

        'RGB_shift_prob': 0.5,
        'shift_limit': (-20, 20),

        'color_jitter_prob': 0.5,
        'brightness': (0.6, 1.3),
        'contrast': (0.6, 1.3),
        'saturation': (0.6, 1.3),
        'hue': (-0.15, 0.15),

        'gaussian_blur_prob': 0.5,
        'blur_limit': (3, 7),
        'sigma_limit': (0.2, 2.0),

        'motion_blur_prob': 0.5,
        'motion_blur_limit': (3, 7),

        'random_erasing_prob': 0.5,
        'random_erasing_mode': 'pixel',
        'random_erasing_min_area': 0.02,
        'random_erasing_max_area': 0.2,
        'random_erasing_max_count': 2,
    }
    def __init__(
        self,
        cfg=_cfg
    ):
        self.transform = A.Compose([
            A.CLAHE(p=cfg['clahe_prob']),
            A.RGBShift(r_shift_limit=cfg['shift_limit'], g_shift_limit=cfg['shift_limit'], b_shift_limit=cfg['shift_limit'], p=cfg['RGB_shift_prob']),
            A.ColorJitter(brightness=cfg['brightness'], contrast=cfg['contrast'], saturation=cfg['saturation'], hue=cfg['hue'], p=cfg['color_jitter_prob']),
            A.AdvancedBlur(blur_limit=cfg['blur_limit'], sigma_x_limit=cfg['sigma_limit'], sigma_y_limit=cfg['sigma_limit'], p=cfg['gaussian_blur_prob']),
            A.MotionBlur(blur_limit=cfg['motion_blur_limit'], p=cfg['motion_blur_prob'], allow_shifted=False),
        ])
        self.random_erasing = RandomErasing(
            probability=cfg['random_erasing_prob'], 
            min_area=cfg['random_erasing_min_area'],
            max_area=cfg['random_erasing_max_area'],
            min_count=cfg['random_erasing_max_count'],
            mode=cfg['random_erasing_mode'], 
            device="cpu")

    #* checked
    def run_color(self, img):
        img = self.transform(image=img)["image"]
        return img

    #* checked
    def run_random_erasing(self, img):
        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 3 # image with (c, h, w) shape
        img = self.random_erasing(img)
        return img

    def _unit_test_color(self, img):
        h, w, _ = img.shape
        img_ls = [img]
        for i in range(63):
            img_i = self.run_color(img)
            img_ls.append(img_i)
        all_img = np.stack(img_ls, 0)
        all_img = all_img.reshape(8, 8, h, w, 3)
        all_img = all_img.transpose(0, 2, 1, 3, 4)
        all_img = all_img.reshape(h*8, w*8, 3)
        cv2.imwrite("tmp.jpg", all_img[..., ::-1])

    def _unit_test_random_erasing(self, img):
        _, h,w = img.shape
        img_ = tensor_to_np(img, is_img=True)
        img_ = inverse_normalize_rgb(img_)
        img_ls = [img_]
        for i in range(63):
            img_i = self.run_random_erasing(img.clone())
            img_i = tensor_to_np(img_i, is_img=True)
            img_i = inverse_normalize_rgb(img_i)
            img_ls.append(img_i)
        all_img = np.stack(img_ls, 0)
        all_img = all_img.reshape(8, 8, h, w, 3)
        all_img = all_img.transpose(0, 2, 1, 3, 4)
        all_img = all_img.reshape(h*8, w*8, 3)
        cv2.imwrite("tmp.jpg", all_img[..., ::-1])


class BaseDataset(Dataset):
    _cfg = {
        "clean_data_mode": "2023_CVPR_HFL",
        "img_size": (640, 480),
        "bbox_scale_factor": 1,
        "patch_size": 256,
        "contact_normal_distance_thresh": (-0.01, 0.01),
        "contact_vertical_distance_thresh": 0.005,
    }

    _aug = {
        "center_jittering": 0.1,
        "scale_factor": 0.2,
        "max_rot": 30,
        "rot_prob": 1,
    }

    def __init__(
        self,
        is_train: bool,
        cfg:Config=_cfg,
        aug:Config=_aug,
    ):
        self.is_train = is_train
        self.aug = aug
        self.cfg = cfg
        self.index_ls = []
        self.image_augmentor = ImageAugmentor()
        self.hm_hand_generator = HeatmapGenerator(self.cfg.heatmap_size, self.cfg.heatmap_hand_sigma)
        self.hm_obj_generator = HeatmapGenerator(self.cfg.heatmap_size, self.cfg.heatmap_obj_sigma)

        self.adp_hm_hand_generator = AdaptiveHeatmapGenerator(21, self.cfg.heatmap_hand_sigma, self.cfg.heatmap_size)
        self.adp_hm_obj_generator = AdaptiveHeatmapGenerator(27, self.cfg.heatmap_obj_sigma, self.cfg.heatmap_size)

    def __len__(self):
        return len(self.index_ls)

    def render_depth(self, verts, faces, cam_intrinsics, background_val=None):
        """ 
            Args:
                verts: (V, 3)
                faces: (F, 3)
                cam_intrinsics: (3, 3)
                img_size: (2, )
            Return:
                depth_maps:     (H, W, 10) np.float64
                pix_to_face:    (H, W, 10) np.uint32
        """
        if not hasattr(self, "renderer"):
            self.renderer = Pytorch3DRenderer(image_size=self.cfg.img_size[::-1], face_per_pixel=20)

        with torch.no_grad():
            verts = torch.from_numpy(verts).float()[None]
            faces = torch.from_numpy(faces).long()[None]
            K = np.zeros([4, 4], dtype=np.float32)
            K[:2, :3] = cam_intrinsics[:2, :3]
            K[[2, 3], [3, 2]] = 1
            K = torch.from_numpy(K).float()[None]
            cameras = self.renderer.create_perspective_cameras(K=K)
            depth_maps, pix_to_face = self.renderer.get_depthMap(verts, faces, cameras)
            depth_maps = depth_maps.squeeze().detach().cpu().numpy()
            pix_to_face = pix_to_face.squeeze().detach().cpu().numpy()
            depth_maps[depth_maps>-1.0] = depth_maps[depth_maps>-1.0] * 1000.0
            if background_val is not None:
                depth_maps[depth_maps==-1.0] = background_val
        return depth_maps, pix_to_face
    
    def get_spatial_aug_params(self, is_train):
        """ Return:
                center_jittering: (2,)
                scale_factor: float
                rot: float
        """
        if is_train:
            center_jittering = self.aug.center_jittering * np.random.uniform(low=-1, high=1, size=2)
            scale_factor = self.aug.scale_factor * np.random.rand() + 1
            if np.random.rand() < self.aug.rot_prob:
                rot = np.random.uniform(low=-1, high=1) * self.aug.max_rot / 180 * np.pi
            else:
                rot = 0
        else:
            center_jittering = np.zeros(2)
            scale_factor = 1
            rot = 0
        return center_jittering, scale_factor, rot

    #* checked
    def get_augmentation_rotmat(self, center_jittering, scale_factor, rot_factor, hand_jt2d, obj_kpt2d, cam_intrinsic):
        """ 
            Args:
                center_jittering: (2,)
                scale_factor: float
                rot: float
                hand_jt2d: (21, 2)
                obj_kpt2d: (27, 2)
            Return:
                rotmat_3d:      (3, 3) np.float64
                rotmat_2d:      (3, 3) np.float64
                cam_intrinsic:  (3, 3) np.float64
        """
        bbox_hand = pt2d_to_bbox2d(hand_jt2d, mode="x1y1x2y2")
        bbox_hand = expand_bbox2d(bbox_hand)
        bbox_hand, max_wh_hand = get_rectanglular_bbox2d(bbox_hand)
        bbox_obj = pt2d_to_bbox2d(obj_kpt2d, mode="x1y1x2y2")
        bbox_obj = expand_bbox2d(bbox_obj)
        bbox_obj, max_wh_obj = get_rectanglular_bbox2d(bbox_obj)
        center = np.concatenate([bbox_hand, bbox_obj], axis=0)
        center = center.reshape(-1, 2).mean(0)


        rotmat_3d = np.array([
            [np.cos(rot_factor), -np.sin(rot_factor), 0],
            [np.sin(rot_factor),  np.cos(rot_factor), 0],
            [          0,            0,               1],
        ])
        
        all_pt2d = np.concatenate([hand_jt2d, obj_kpt2d], axis=0)
        # center = (np.max(all_pt2d, axis=0) + np.min(all_pt2d, axis=0)) / 2
        # center = np.mean(all_pt2d, axis=0) + center_jittering
        raduis = np.max(np.linalg.norm(all_pt2d - center, axis=-1))
        center = center + center_jittering * raduis
        raduis = raduis * self.cfg.bbox_scale_factor * scale_factor
        scale = self.cfg.patch_size / (raduis * 2)
        center_rot = center @ rotmat_3d[:2, :2].T * scale
        t = np.array([self.cfg.patch_size // 2, self.cfg.patch_size // 2]) + .5 - center_rot
        St_2d = np.array([
            [scale,     0,  t[0]],
            [    0, scale,  t[1]],
            [    0,     0,     1],
        ])
        rotmat_2d = St_2d @ rotmat_3d # affine transformation matrix

        #! 3D rotation around camera optical axis is the indentical to 2D rotation around the principal point
        center_rot_ = center - cam_intrinsic[:2, 2]  
        center_rot_ = center_rot_ @ rotmat_3d[:2, :2].T * scale
        t_ = np.array([self.cfg.patch_size // 2, self.cfg.patch_size // 2]) + .5 - center_rot_
        cam_intrinsic_corp = cam_intrinsic.copy()
        cam_intrinsic_corp[:2] *= scale
        cam_intrinsic_corp[:2, 2] = t_
        return rotmat_3d, rotmat_2d, cam_intrinsic_corp
    

    def get_augmentation_rotmat_tmp(self, center_jittering, scale_factor, rot_factor, hand_jt2d, obj_kpt2d, cam_intrinsic):
        """ 
            Args:
                center_jittering: (2,)
                scale_factor: float
                rot: float
                hand_jt2d: (21, 2)
                obj_kpt2d: (27, 2)
            Return:
                rotmat_3d:      (3, 3) np.float64
                rotmat_2d:      (3, 3) np.float64
                cam_intrinsic:  (3, 3) np.float64
        """
        bbox_hand = pt2d_to_bbox2d(hand_jt2d, mode="x1y1x2y2")
        bbox_hand = expand_bbox2d(bbox_hand)
        # bbox_hand, max_wh_hand = get_rectanglular_bbox2d(bbox_hand)
        bbox_obj = pt2d_to_bbox2d(obj_kpt2d, mode="x1y1x2y2")
        bbox_obj = expand_bbox2d(bbox_obj)
        # bbox_obj, max_wh_obj = get_rectanglular_bbox2d(bbox_obj)
        center = np.concatenate([bbox_hand, bbox_obj], axis=0)
        center = center.reshape(-1, 2)
        center = np.stack([center.min(0), center.max(0)]).mean(0)

        rotmat_3d = np.array([
            [np.cos(rot_factor), -np.sin(rot_factor), 0],
            [np.sin(rot_factor),  np.cos(rot_factor), 0],
            [          0,            0,               1],
        ])
        
        all_pt2d = np.concatenate([hand_jt2d, obj_kpt2d], axis=0)
        # center = (np.max(all_pt2d, axis=0) + np.min(all_pt2d, axis=0)) / 2
        # center = np.mean(all_pt2d, axis=0) + center_jittering
        raduis = np.max(np.linalg.norm(all_pt2d - center, axis=-1))
        center = center + center_jittering * raduis
        raduis = raduis * self.cfg.bbox_scale_factor * scale_factor
        scale = self.cfg.patch_size / (raduis * 2)
        center_rot = center @ rotmat_3d[:2, :2].T * scale
        t = np.array([self.cfg.patch_size // 2, self.cfg.patch_size // 2]) + .5 - center_rot
        St_2d = np.array([
            [scale,     0,  t[0]],
            [    0, scale,  t[1]],
            [    0,     0,     1],
        ])
        rotmat_2d = St_2d @ rotmat_3d # affine transformation matrix

        #! 3D rotation around camera optical axis is the indentical to 2D rotation around the principal point
        center_rot_ = center - cam_intrinsic[:2, 2]  
        center_rot_ = center_rot_ @ rotmat_3d[:2, :2].T * scale
        t_ = np.array([self.cfg.patch_size // 2, self.cfg.patch_size // 2]) + .5 - center_rot_
        cam_intrinsic_corp = cam_intrinsic.copy()
        cam_intrinsic_corp[:2] *= scale
        cam_intrinsic_corp[:2, 2] = t_
        return rotmat_3d, rotmat_2d, cam_intrinsic_corp
    
    #* checked
    def get_obj_front_and_back_depth_map(self, obj_name, obj_6D, cam_intrinsics, color_file, background_val):
        """ Load object depth map and pixel_to_face_indices map from cache path if it exist, otherwise render them using pytorch3d. 
            args:
                verts: (V, 3)
                faces: (N, 3)
                cam_intrinsics: (3, 3)
                color_file: str
            return: 
                depth_front:    (H, W) np.int64
                depth_back:     (H, W) np.int64
                front_face_map: (H, W) np.int64
                back_face_map:  (H, W) np.int64
        """
        file_name = color_file.replace("color_", "depth_").replace(".jpg", ".png")
        cache_path_front = os.path.join(self.data_dir, "cache", "obj_depth_map", "front", file_name)
        cache_path_back = os.path.join(self.data_dir, "cache", "obj_depth_map", "back", file_name)
        cache_path_front_face_map = os.path.join(self.data_dir, "cache", "obj_depth_map", "front_face_map", file_name)
        cache_path_back_face_map = os.path.join(self.data_dir, "cache", "obj_depth_map", "back_face_map", file_name)
        if os.path.exists(cache_path_front) and os.path.exists(cache_path_back) and \
            os.path.exists(cache_path_front_face_map) and os.path.exists(cache_path_back_face_map):
            depth_front = dep_to_3channel_inv(cv2.imread(cache_path_front))
            depth_back = dep_to_3channel_inv(cv2.imread(cache_path_back))
            front_face_map = dep_to_3channel_inv(cv2.imread(cache_path_front_face_map))
            back_face_map = dep_to_3channel_inv(cv2.imread(cache_path_back_face_map))
            front_face_map[front_face_map == 256**3 - 1] = -1
            back_face_map[back_face_map == 256**3 - 1] = -1
        else:
            verts = YCB_MESHES[obj_name]["verts"]
            verts = verts @ obj_6D[:3, :3].T + obj_6D[:3, 3]
            faces = YCB_MESHES[obj_name]["faces"]
            depth_maps, pix_to_face = self.render_depth(verts, faces, cam_intrinsics, background_val=0.0)
            depth_front = depth_maps[..., 0]
            front_face_map = pix_to_face[..., 0]
            depth_maps # (480, 640, 10)
            back_arg = depth_maps[..., 1:].argmax(-1) # (480, 640)
            XX = np.arange(pix_to_face.shape[0])
            YY = np.arange(pix_to_face.shape[1])
            XX, YY = np.meshgrid(XX, YY, indexing='ij')
            depth_back = depth_maps[..., 1:][XX, YY, back_arg]  #! get the max depth formatting the back depth map
            back_face_map = pix_to_face[..., 1:][XX, YY, back_arg]

            os.makedirs(os.path.dirname(cache_path_front), exist_ok=True)
            os.makedirs(os.path.dirname(cache_path_back), exist_ok=True)
            os.makedirs(os.path.dirname(cache_path_front_face_map), exist_ok=True)
            os.makedirs(os.path.dirname(cache_path_back_face_map), exist_ok=True)
            cv2.imwrite(cache_path_front, dep_to_3channel(depth_front))
            cv2.imwrite(cache_path_back, dep_to_3channel(depth_back))
            front_face_map_save = front_face_map.copy()
            front_face_map_save[front_face_map_save == -1] = 256**3 - 1
            cv2.imwrite(cache_path_front_face_map, dep_to_3channel(front_face_map_save))
            back_face_map_save = back_face_map.copy()
            back_face_map_save[back_face_map_save == -1] = 256**3 - 1
            cv2.imwrite(cache_path_back_face_map, dep_to_3channel(back_face_map_save))

        depth_front[depth_front == 0.0] = background_val
        depth_back[depth_back == 0.0] = background_val
        return depth_front, depth_back, front_face_map, back_face_map

    def get_color_aug_params(self, is_train):
        if is_train:
            ...

    def get_hand_and_object_contact(
        self,
        mano_pose_aa_flat,
        mano_beta,
        mano_global_rot,
        mano_global_transl,
        is_right,
        obj_name,
        obj_6D,
        front_face_ind_map,
        back_face_ind_map,
        color_file,
    ):
        """ 
            Args: 
                mano_pose_aa_flat: (45,)
                mano_beta: (10,)
                mano_global_rot: (3,)
                mano_global_transl: (3,)
                is_right: bool
                obj_name: str
                obj_6D: (3, 4)
                front_face_ind_map: (H, W) np.int64
                back_face_ind_map: (H, W) np.int64
                color_file: str
            Return:
                hand_contact: (V,) np.float64
                front_map_to_joint: (H, W) np.int32
                front_contact_weight_map: (H, W) np.float64
                back_map_to_joint: (H, W) np.int32
                back_contact_weight_map: (H, W) np.float64
        
        """
        file_name = color_file.replace("color_", "contact_")
        cache_path_hand = os.path.join(self.data_dir, "cache", "hand_contact", file_name.replace(".jpg", ".npy"))
        cache_path_obj_front_joint = os.path.join(self.data_dir, "cache", "obj_contact", "front_joint", file_name.replace(".jpg", ".png"))
        cache_path_obj_front_weight = os.path.join(self.data_dir, "cache", "obj_contact", "front_weight", file_name.replace(".jpg", ".png"))
        cache_path_obj_back_joint = os.path.join(self.data_dir, "cache", "obj_contact", "back_joint", file_name.replace(".jpg", ".png"))
        cache_path_obj_back_weight = os.path.join(self.data_dir, "cache", "obj_contact", "back_weight", file_name.replace(".jpg", ".png"))
        if os.path.exists(cache_path_hand) and os.path.exists(cache_path_obj_front_joint) and \
            os.path.exists(cache_path_obj_front_weight) and os.path.exists(cache_path_obj_back_joint) and os.path.exists(cache_path_obj_back_weight):
            hand_contact = np.load(cache_path_hand)
            front_map_to_joint = cv2.imread(cache_path_obj_front_joint, cv2.IMREAD_GRAYSCALE).astype(np.int32)
            front_map_to_joint[front_map_to_joint == 255] = -1
            front_contact_weight_map = dep_to_3channel_inv(cv2.imread(cache_path_obj_front_weight)).astype(np.float64) / (256**3 - 1)
            back_map_to_joint = cv2.imread(cache_path_obj_back_joint, cv2.IMREAD_GRAYSCALE).astype(np.int32)
            back_map_to_joint[back_map_to_joint == 255] = -1
            back_contact_weight_map = dep_to_3channel_inv(cv2.imread(cache_path_obj_back_weight)).astype(np.float64) / (256**3 - 1)
        else:
            hand_verts, hand_jt3d = get_hand_vert(mano_pose_aa_flat, mano_beta, mano_global_rot, mano_global_transl, is_right=is_right)
            hand_faces = mano_layer_r.th_faces if is_right else mano_layer_l.th_faces
            hand_faces = hand_faces.squeeze().detach().cpu().numpy()
            hand_mesh = trimesh.Trimesh(hand_verts, hand_faces)
            hand_normals = hand_mesh.vertex_normals
            hand_normals = hand_normals / (np.linalg.norm(hand_normals, axis=-1, keepdims=True) + 1e-20)
            hand_verts_filled, hand_normals_filled = fill_finger_gaps_in_mano(hand_verts, hand_normals)
            hand_normals_filled = hand_normals_filled / (np.linalg.norm(hand_normals_filled, axis=-1, keepdims=True) + 1e-20)


            obj_verts_ori = YCB_MESHES[obj_name]["verts"]
            obj_verts_transf = obj_verts_ori @ obj_6D[:3, :3].T + obj_6D[:3, 3]
            obj_normals_transf = YCB_MESHES[obj_name]["normals"] @ obj_6D[:3, :3].T

            hand_contact, obj_contact, obj_contact_to_hand_vert = detect_hand_and_object_contact(
                hand_verts_filled, 
                hand_normals_filled, 
                obj_verts_transf, 
                obj_normals_transf, 
                normal_distance_thresh=self.cfg.contact_normal_distance_thresh,
                vertical_distance_thresh=self.cfg.contact_vertical_distance_thresh
            )


            vert2finger = np.zeros(hand_verts_filled.shape[0], dtype=np.int32)
            for k, v in FINGER_VERTS_IDX.items():
                vert2finger[v] = FINGER_IDX[k]

            # region [generate front & back object contact map]
            front_face_mask = front_face_ind_map != -1
            front_face_map = front_face_ind_map[..., None].repeat(3, axis=-1)
            front_face_map[front_face_mask] = YCB_MESHES[obj_name]["faces"][front_face_map[..., 0][front_face_mask]].copy()
            front_face_obj_mask = front_face_map != -1

            front_contact_to_joint = front_face_map.copy()
            front_contact_to_joint[front_face_obj_mask] = obj_contact_to_hand_vert[front_face_map[front_face_obj_mask]]
            front_contact_to_joint_mask = front_contact_to_joint != -1
            front_contact_to_joint[front_contact_to_joint_mask] = vert2finger[front_contact_to_joint[front_contact_to_joint_mask]]
            contact_pixels_mask = front_contact_to_joint.sum(-1) != -3
            front_contact_to_joint[contact_pixels_mask] = np.sort(front_contact_to_joint[contact_pixels_mask], axis=-1)
            front_map_to_joint = front_contact_to_joint[..., 1] #* the middle index is the most frequent index

            front_contact_weight_map = front_face_map.copy().astype(np.float64)
            front_contact_weight_map[front_face_obj_mask] = obj_contact[front_face_map[front_face_obj_mask]]
            front_contact_weight_map[~front_face_obj_mask] = 0
            front_contact_weight_map = front_contact_weight_map.mean(-1)


            back_face_mask = back_face_ind_map != -1
            back_face_map = back_face_ind_map[..., None].repeat(3, axis=-1)
            back_face_map[back_face_mask] = YCB_MESHES[obj_name]["faces"][back_face_map[..., 0][back_face_mask]].copy()
            back_face_obj_mask = back_face_map != -1

            back_contact_to_joint = back_face_map.copy()
            back_contact_to_joint[back_face_obj_mask] = obj_contact_to_hand_vert[back_face_map[back_face_obj_mask]]
            back_contact_to_joint_mask = back_contact_to_joint != -1
            back_contact_to_joint[back_contact_to_joint_mask] = vert2finger[back_contact_to_joint[back_contact_to_joint_mask]]
            contact_pixels_mask = back_contact_to_joint.sum(-1) != -3
            back_contact_to_joint[contact_pixels_mask] = np.sort(back_contact_to_joint[contact_pixels_mask], axis=-1)
            back_map_to_joint = back_contact_to_joint[..., 1]

            back_contact_weight_map = back_face_map.copy().astype(np.float64)
            back_contact_weight_map[back_face_obj_mask] = obj_contact[back_face_map[back_face_obj_mask]]
            back_contact_weight_map[~back_face_obj_mask] = 0
            back_contact_weight_map = back_contact_weight_map.mean(-1)
            # endregion

            
            # region [save cache]
            os.makedirs(os.path.dirname(cache_path_hand), exist_ok=True)
            np.save(cache_path_hand, hand_contact)
            
            front_map_to_joint_save = front_map_to_joint.copy()
            front_map_to_joint_save[front_map_to_joint_save == -1] = 255
            front_map_to_joint_save = front_map_to_joint_save.astype(np.uint8)
            os.makedirs(os.path.dirname(cache_path_obj_front_joint), exist_ok=True)
            cv2.imwrite(cache_path_obj_front_joint, front_map_to_joint_save)

            front_contact_weight_map_ = front_contact_weight_map.copy() * (256**3 - 1)
            front_contact_weight_map_save = dep_to_3channel(front_contact_weight_map_)
            os.makedirs(os.path.dirname(cache_path_obj_front_weight), exist_ok=True)
            cv2.imwrite(cache_path_obj_front_weight, front_contact_weight_map_save)

            back_map_to_joint_save = back_map_to_joint.copy()
            back_map_to_joint_save[back_map_to_joint_save == -1] = 255
            back_map_to_joint_save = back_map_to_joint_save.astype(np.uint8)
            os.makedirs(os.path.dirname(cache_path_obj_back_joint), exist_ok=True)
            cv2.imwrite(cache_path_obj_back_joint, back_map_to_joint_save)

            back_contact_weight_map_ = back_contact_weight_map.copy() * (256**3 - 1)
            back_contact_weight_map_save = dep_to_3channel(back_contact_weight_map_)
            os.makedirs(os.path.dirname(cache_path_obj_back_weight), exist_ok=True)
            cv2.imwrite(cache_path_obj_back_weight, back_contact_weight_map_save)
            # endregion

        return hand_contact, front_map_to_joint, front_contact_weight_map, back_map_to_joint, back_contact_weight_map


    def get_hand_contact(
        self,
        mano_pose_aa_flat,
        mano_beta,
        mano_global_rot,
        mano_global_transl,
        is_right,
        obj_name,
        obj_6D,
        color_file,
    ):
        """ 
            Args: 
                mano_pose_aa_flat: (45,)
                mano_beta: (10,)
                mano_global_rot: (3,)
                mano_global_transl: (3,)
                is_right: bool
                obj_name: str
                obj_6D: (3, 4)
                front_face_ind_map: (H, W) np.int64
                back_face_ind_map: (H, W) np.int64
                color_file: str
            Return:
                hand_contact: (V,) np.float64
                front_map_to_joint: (H, W) np.int32
                front_contact_weight_map: (H, W) np.float64
                back_map_to_joint: (H, W) np.int32
                back_contact_weight_map: (H, W) np.float64
        
        """
        if "DexYCB" in self.data_dir:
            file_name = color_file.replace("color_", "contact_")
            cache_path_hand = os.path.join(self.data_dir, "cache", "hand_contact", file_name.replace(".jpg", ".npy"))
        elif 'HO3D_v2' in self.data_dir:
            file_name = color_file.replace("meta", "hand_contact") + '.npy'
            cache_path_hand = os.path.join(self.data_dir, "cache", "hand_contact", file_name)
        else:
            raise NotImplementedError

        if os.path.exists(cache_path_hand):
            hand_contact = np.load(cache_path_hand)
        else:
            hand_verts, hand_jt3d = get_hand_vert(mano_pose_aa_flat, mano_beta, mano_global_rot, mano_global_transl, is_right=is_right)
            hand_faces = mano_layer_r.th_faces if is_right else mano_layer_l.th_faces
            hand_faces = hand_faces.squeeze().detach().cpu().numpy()
            hand_mesh = trimesh.Trimesh(hand_verts, hand_faces)
            hand_normals = hand_mesh.vertex_normals
            hand_normals = hand_normals / (np.linalg.norm(hand_normals, axis=-1, keepdims=True) + 1e-20)
            hand_verts_filled, hand_normals_filled = fill_finger_gaps_in_mano(hand_verts, hand_normals)
            hand_normals_filled = hand_normals_filled / (np.linalg.norm(hand_normals_filled, axis=-1, keepdims=True) + 1e-20)


            obj_verts_ori = YCB_MESHES[obj_name]["verts"]
            obj_verts_transf = obj_verts_ori @ obj_6D[:3, :3].T + obj_6D[:3, 3]
            obj_normals_transf = YCB_MESHES[obj_name]["normals"] @ obj_6D[:3, :3].T

            hand_contact, obj_contact, obj_contact_to_hand_vert = detect_hand_and_object_contact(
                hand_verts_filled, 
                hand_normals_filled, 
                obj_verts_transf, 
                obj_normals_transf, 
                normal_distance_thresh=self.cfg.contact_normal_distance_thresh,
                vertical_distance_thresh=self.cfg.contact_vertical_distance_thresh
            )
            
            # region [save cache]
            os.makedirs(os.path.dirname(cache_path_hand), exist_ok=True)
            np.save(cache_path_hand, hand_contact)
            # endregion

        return hand_contact
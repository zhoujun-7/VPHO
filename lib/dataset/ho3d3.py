import os
import json
import time
import pickle
from natsort import natsorted
import numpy as np
import torch
import cv2
import yaml
import tqdm
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d

from lib.dataset.base import (BaseDataset,
                              YCB_CLASSES, 
                              YCB_ID,
                              YCB_MESHES,
                              get_hand_vert,
                              mano_layer_r, 
                              mano_layer_l, 
                              filter_DexYCB_by_HFL, 
                              filter_DexYCB_by_ArtiBoost, 
                              Filter_DexYCB_by_Steady_Grasping,
                              np_to_tensor,
                              normalize_rgb,
                              normalize_depth,
                              normalize_contact,
                              )
from lib.configs.args import Config
from lib.utils.transform_fn import project_pt3d_to_pt2d, OPENGL_TO_OPENCV
from lib.utils.misc_fn import (
    pt2d_to_bbox2d, expand_bbox2d, check_bbox2d, get_rectanglular_bbox2d, get_inter_bbox2d, get_unite_bbox2d, fuse_bbox, bx2d2_to_bx2d4, AdaptiveHeatmapGenerator
)
from lib.utils.physics_fn import VERT2ANCHOR
from lib.utils.viz_fn import draw_pts_on_image, draw_bbox_on_image
from lib.utils.hand_fn import joint_reorder, get_joint_aligned_with_HO3D
from lib.utils.physics_fn import VERT2ANCHOR

class HO3DDataset_Force(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        is_train: bool,
        aug=BaseDataset._aug,
        cfg=BaseDataset._cfg,
    ):
        self.data_dir = data_dir
        super().__init__(is_train, aug, cfg)
        self.load_gravity()
        self.load_is_grasped()

    def load_samples(self, data_dir, is_trainset=True):
        split = "train" if is_trainset else "evaluation"
        split_dir = os.path.join(data_dir, split)
        subj_dir = natsorted([os.path.join(split_dir, d) for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        meta_dir = [os.path.join(d, "meta") for d in subj_dir]
        index_ls = []
        for subj in natsorted(os.listdir(split_dir)):
            subj_dir = os.path.join(split_dir, subj)
            meta_dir = os.path.join(subj_dir, "meta")
            if os.path.isdir(meta_dir):
                for anno in natsorted(os.listdir(meta_dir)):
                    anno = os.path.splitext(anno)[0]
                    index_ls.append(os.path.join(meta_dir, anno))
        return index_ls
    
    def load_gravity(self, path='asset/ours/HO3D_v2/gravity_direction.json'):
        with open(path, 'r') as f:
            self.dir2gravity = json.load(f)
    
    def get_gravity(self, filename):
        g_key = filename.split('/meta')[0].split('/')[-1]
        if g_key in self.dir2gravity:
            return self.dir2gravity[g_key]
        else:
            return np.array([0, 1, 0])
        
    def load_is_grasped(self, path="asset/ours/HO3D_v2/is_off_desk.pkl"):
        with open(path, 'rb') as f:
            self.is_grasped = pickle.load(f)
    
    def get_is_grasped(self, filename):
        is_g_key = filename.split('/')
        ind = int(is_g_key[-1])
        is_g_key = f"{is_g_key[-4]}/{is_g_key[-3]}"
        v = self.is_grasped[is_g_key]
        is_grasp = v[ind]
        return is_grasp

    def get_force(self, filename):
        force_path = filename.replace("HO3D_v2/", "HO3D_v2/cache/hand_force/").replace('.png', '.pkl').replace('rgb/', 'hand_force/')
        cache_path = os.path.join(self.data_dir, "cache", "hand_force", force_path)
        with open(cache_path, 'rb') as f: force_dt = pickle.load(f)
        force_local = force_dt['force_local']
        force_global = force_dt['force_global']
        return force_local, force_global
    
    def __getitem__(self, index):
        ...

    def get_train_item(self, index):
        sample_path = self.index_ls[index]
        with open(sample_path+'.pkl', 'rb') as f:
            sample = pickle.load(f)
            
        rgb_split = sample_path.split("/")
        rgb_split[-2] = "rgb"
        rgb_path = sample_path.replace("meta", "rgb") + ".png"
        rgb = cv2.imread(rgb_path)[..., ::-1].copy()
        cam_intrinsic = sample["camMat"]

        # region [load hand], checked
        is_right = True #! only right hand in HO3D_v2
        pose_m = sample["handPose"]
        mano_beta = sample["handBeta"]
        jt3d_GL = sample["handJoints3D"] #! in OpenGL coordinate system, ManoLayer order, definitions of tips are different from Manopth 
        jt3d = jt3d_GL @ OPENGL_TO_OPENCV.T
        mano_global_rot = pose_m[:3]
        mano_global_transl = sample["handTrans"]
        mano_pose_aa_flat = pose_m[3:]

        mano_global_rot_mat = axis_angle_to_matrix(torch.from_numpy(mano_global_rot)).numpy()
        mano_global_rot_mat = OPENGL_TO_OPENCV @ mano_global_rot_mat
        mano_global_rot = matrix_to_axis_angle(torch.from_numpy(mano_global_rot_mat)).numpy()
        mano_global_transl = OPENGL_TO_OPENCV @ sample["handTrans"]

        vert3d, _jt3d = get_hand_vert(mano_pose_aa_flat, mano_beta, mano_global_rot, mano_global_transl, is_right) #! _jt3d != jt3d, as the definitions of jt3d are different. However, they are the same in term of hand pose.
        trans_offset = jt3d[0] - _jt3d[0]
        mano_global_transl = mano_global_transl + trans_offset
        vert3d, _jt3d = get_hand_vert(mano_pose_aa_flat, mano_beta, mano_global_rot, mano_global_transl, is_right)
        jt2d = project_pt3d_to_pt2d(_jt3d, cam_intrinsic)
        _jt3d = get_joint_aligned_with_HO3D(vert3d, _jt3d) #* checked, vert3d and _jt3d are the same with annotations
        # endregion

        # region [load object], checked
        obj_rot_aa = torch.from_numpy(sample["objRot"]).squeeze()
        obj_rot_mat = axis_angle_to_matrix(obj_rot_aa).numpy()
        obj_trans = sample["objTrans"]
        obj_6D = np.concatenate([obj_rot_mat, obj_trans[:, None]], axis=1)
        obj_name = sample["objName"]
        obj_id = YCB_ID[obj_name]
        obj_verts_ori = YCB_MESHES[obj_name]["verts_sampled"]
        
        obj_6D = OPENGL_TO_OPENCV @ obj_6D
        # obj_verts_cam = obj_verts_ori @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        obj_kpt3d = YCB_MESHES[obj_name]["kpt3d"] @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        obj_kpt2d = project_pt3d_to_pt2d(obj_kpt3d, cam_intrinsic)
        obj_CoM = YCB_MESHES[obj_name]["CoM"] @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        # endregion

        # physics
        gravity = self.get_gravity(sample_path)
        gravity = np.array(gravity)
        gravity = gravity / np.linalg.norm(gravity)
        
        hand_contact = self.get_hand_contact(
            mano_pose_aa_flat,
            mano_beta,
            mano_global_rot,
            mano_global_transl,
            is_right,
            obj_name,
            obj_6D,
            sample_path.replace(self.data_dir+'/', ""),
        )
        force_contact = VERT2ANCHOR.get_force_contact(hand_contact)
        is_grasped = VERT2ANCHOR.check_is_grasped(force_contact)
        force_local, force_global = self.get_force(rgb_path)

        # region [get spatial augmentation data]
        center_jittering, scale_factor, rot_factor = self.get_spatial_aug_params(self.is_train)
        n = 100
        while n:= n - 1:
            rotmat_3d, rotmat_2d, cam_intrinsic_crop = self.get_augmentation_rotmat(center_jittering, scale_factor, rot_factor, jt2d, obj_kpt2d, cam_intrinsic)
        # region [bbox2d] #! make sure bbox in the image
            rgb_patch = cv2.warpAffine(rgb, rotmat_2d[:2, :], (self.cfg.patch_size, self.cfg.patch_size), flags=cv2.INTER_CUBIC)
            _jt2d = jt2d @ rotmat_2d[:2, :2].T + rotmat_2d[:2, 2]
            _obj_kpt2d = obj_kpt2d @ rotmat_2d[:2, :2].T + rotmat_2d[:2, 2]
            bbox_hand = pt2d_to_bbox2d(_jt2d, mode="x1y1x2y2")
            bbox_hand = expand_bbox2d(bbox_hand, scale_factor=1.15)
            bbox_hand_rect, max_wh_hand = get_rectanglular_bbox2d(bbox_hand)
            is_ok1 = check_bbox2d(bbox_hand_rect, rgb_patch)
            bbox_obj = pt2d_to_bbox2d(_obj_kpt2d, mode="x1y1x2y2")
            bbox_obj = expand_bbox2d(bbox_obj, scale_factor=1.10)
            bbox_obj_rect, max_wh_obj = get_rectanglular_bbox2d(bbox_obj)
            is_ok2 = check_bbox2d(bbox_obj_rect, rgb_patch)
            if is_ok1 and is_ok2:
                break
            else:
                scale_factor *= 1.01
        if n == 0:
            raise ValueError(f"index {index} bbox out of image. {rgb_path}")
        jt2d, obj_kpt2d = _jt2d, _obj_kpt2d
        # endregion[bbox2d]

        # region [do 3D spatial augmentation]
        jt3d = jt3d @ rotmat_3d.T
        mano_global_rotmat = axis_angle_to_matrix(torch.tensor(mano_global_rot)).numpy()
        mano_global_rotmat = rotmat_3d @ mano_global_rotmat
        mano_global_rot = matrix_to_axis_angle(torch.tensor(mano_global_rotmat)).numpy()
        gt_hand_vert, _jt3d = get_hand_vert(mano_pose_aa_flat, mano_beta, mano_global_rot, mano_global_transl, is_right)
        mano_global_transl = mano_global_transl + (jt3d[0] - _jt3d[0]) #! correct translation
        gt_hand_vert = gt_hand_vert + (jt3d[0] - _jt3d[0])
        _jt3d = _jt3d + (jt3d[0] - _jt3d[0])

        obj_6D[:3, :3] = rotmat_3d @ obj_6D[:3, :3]
        obj_6D[:3, 3] = rotmat_3d @ obj_6D[:3, 3]
        obj_kpt3d = YCB_MESHES[obj_name]["kpt3d"] @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        # obj_kpt2d_proj = project_pt3d_to_pt2d(obj_kpt3d, cam_intrinsic_crop) # indentical to obj_kpt2d
        # obj_verts_cam = obj_verts_ori @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        obj_CoM = obj_CoM @ rotmat_3d.T
        gravity = gravity @ rotmat_3d.T
        # endregion

        # region [color augmentation]  #* checked
        if self.is_train:
            rgb_patch = self.image_augmentor.run_color(rgb_patch)
        # endregion

        # region [flip left hand] #* HO3D_v2 only has right hand
        gt_hand_vert_flip = gt_hand_vert.copy()
        gt_hand_jt3d_flip = jt3d.copy()
        cam_intrinsic_crop_flip = cam_intrinsic_crop.copy()
        if not is_right:
            ...

        gt_hand_vert_flip = gt_hand_vert_flip - gt_hand_jt3d_flip[0]
        gt_hand_jt3d_flip = gt_hand_jt3d_flip - gt_hand_jt3d_flip[0]
        # endregion

        # region [Heatmap]
        hm_hand = self.adp_hm_hand_generator(jt2d, bbox_hand)
        hm_obj = self.hm_hand_generator.get_heatmap(obj_kpt2d, bbox_obj_rect)
        # endregion

        # region [normalize data]
        rgb_normalized = normalize_rgb(rgb_patch)
        rgb_tensor = np_to_tensor(rgb_normalized, is_img=True)
        if self.is_train:
            rgb_tensor = self.image_augmentor.run_random_erasing(rgb_tensor) #* checked, random erasing augmentation

        root_joint = jt3d[0]
        root_joint = torch.from_numpy(root_joint)
        obj_6D = torch.from_numpy(obj_6D)
        obj_6D[:3, 3] = obj_6D[:3, 3] - root_joint #* to translation raletive to hand wrist
        obj_verts_cam = obj_verts_ori @ obj_6D[:3, :3].T.numpy() + obj_6D[:3, 3].numpy()
        obj_rot6d = matrix_to_rotation_6d(obj_6D[:3, :3])
        obj_pose = torch.cat([obj_rot6d, obj_6D[:3, 3]], dim=-1)
        # obj_pose = torch.from_numpy(obj_pose)
        mano_params = np.concatenate([mano_global_rot, mano_pose_aa_flat, mano_beta]) # (58,) #* checked, use mean pose instead of flat pose
        mano_params = torch.from_numpy(mano_params)
        root_joint_flip = _jt3d[0]
        root_joint_flip = torch.from_numpy(root_joint_flip)

        obj_CoM = torch.from_numpy(obj_CoM)
        obj_CoM = obj_CoM - root_joint

        cam_intrinsic = torch.from_numpy(cam_intrinsic)
        cam_intrinsic_crop = torch.from_numpy(cam_intrinsic_crop)
        jt3d = torch.from_numpy(jt3d)
        force_point, _ = VERT2ANCHOR(gt_hand_vert_flip)

        gravity = torch.from_numpy(gravity).to(torch.float32)

        # endregion

        out = {
            "is_ho3d": True,
            "index": index,
            "rgb_path": rgb_path,
            "rgb": rgb_tensor,
            "root_joint": root_joint,
            "bbox_hand": bbox_hand,
            "bbox_obj": bbox_obj,
            "bbox_hand_rect": bbox_hand_rect,
            "bbox_obj_rect": bbox_obj_rect,

            "hm_hand": hm_hand,
            "hm_obj": hm_obj,
            "is_right": is_right,
            "gt_jt2d": jt2d,
            "gt_obj2d": obj_kpt2d,

            "gt_obj": obj_pose,
            "gt_mano": mano_params, # flipped, include error in mano shape
            "gt_joint": jt3d, # never flipped, identical to annotation
            "gt_hand_vert": gt_hand_vert, # never flipped, identical to annotation
            "gt_hand_jt3d_flip": gt_hand_jt3d_flip, # flipped, for supervision
            "gt_hand_vert_flip": gt_hand_vert_flip, # flipped, for supervision
            "root_joint_flip": root_joint_flip,
            "obj_name": obj_name,
            "obj_id": obj_id-1,
            "cam_intr": cam_intrinsic,
            "cam_intr_crop": cam_intrinsic_crop,
            "cam_intr_crop_flip": cam_intrinsic_crop_flip,

            "gravity": gravity[None],
            "obj_CoM": obj_CoM[None],
            "is_grasped": is_grasped,
            "force_contact": force_contact,
            "force_local": force_local,
            "force_global": force_global,
            "force_point": force_point,
        }
        return out
    
    def get_eval_item(self, index):
        sample_path = self.index_ls[index]
        with open(sample_path+'.pkl', 'rb') as f:
            sample = pickle.load(f)

        is_right = True #! only right hand in HO3D_v2

        rgb_split = sample_path.split("/")
        rgb_split[-2] = "rgb"
        rgb_path = sample_path.replace("meta", "rgb") + ".png"
        rgb = cv2.imread(rgb_path)[..., ::-1].copy()
        cam_intrinsic = sample["camMat"]

        root_joint = sample["handJoints3D"] # in m
        root_joint = root_joint @ OPENGL_TO_OPENCV.T
        bbox_hand = np.array(sample["handBoundingBox"])

        obj_name = sample["objName"]
        obj_id = YCB_ID[obj_name]
        obj_rot_aa = sample["objRot"].squeeze()
        obj_rot_mat = axis_angle_to_matrix(torch.from_numpy(obj_rot_aa)).numpy()
        obj_trans = sample["objTrans"]
        obj_6D = np.concatenate([obj_rot_mat, obj_trans[:, None]], axis=1)
        obj_6D = OPENGL_TO_OPENCV @ obj_6D

        obj_kpt3d = YCB_MESHES[obj_name]["kpt3d_axsym"] @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        obj_kpt2d = project_pt3d_to_pt2d(obj_kpt3d, cam_intrinsic)
        obj_CoM = YCB_MESHES[obj_name]["CoM"] @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        obj_CoM_z = obj_CoM[2] * 1000 # in mm

        center_jittering, scale_factor, rot_factor = self.get_spatial_aug_params(self.is_train)
        n = 100 # 1.01 ** 100 = 2.7
        while n:= n - 1:
            _jt2d = bx2d2_to_bx2d4(bbox_hand.reshape(-1, 2))
            rotmat_3d, rotmat_2d, cam_intrinsic_crop = self.get_augmentation_rotmat(center_jittering, scale_factor, rot_factor, _jt2d, obj_kpt2d, cam_intrinsic)
            rgb_patch = cv2.warpAffine(rgb, rotmat_2d[:2, :], (self.cfg.patch_size, self.cfg.patch_size), flags=cv2.INTER_CUBIC)
            bbox_hand = (bbox_hand.reshape(-1, 2) @ rotmat_2d[:2, :2].T + rotmat_2d[:2, 2]).reshape(-1)
            _obj_kpt2d = obj_kpt2d @ rotmat_2d[:2, :2].T + rotmat_2d[:2, 2]
            bbox_hand = expand_bbox2d(bbox_hand, scale_factor=1.2)
            bbox_hand_rect, max_wh_hand = get_rectanglular_bbox2d(bbox_hand)
            is_ok1 = check_bbox2d(bbox_hand_rect, rgb_patch)
            bbox_obj = pt2d_to_bbox2d(_obj_kpt2d, mode="x1y1x2y2")
            bbox_obj = expand_bbox2d(bbox_obj, scale_factor=1.00)
            bbox_obj_rect, max_wh_obj = get_rectanglular_bbox2d(bbox_obj)
            is_ok2 = check_bbox2d(bbox_obj_rect, rgb_patch)
            if is_ok1 and is_ok2:
                break
            else:
                scale_factor *= 1.01
        if n == 0:
            raise ValueError(f"index {index} bbox out of image")
        obj_kpt2d = _obj_kpt2d


        # rot object. #* checked
        obj_6D[:3, :3] = rotmat_3d @ obj_6D[:3, :3]
        obj_6D[:3, 3] = rotmat_3d @ obj_6D[:3, 3]
        obj_kpt3d = YCB_MESHES[obj_name]["kpt3d"] @ obj_6D[:3, :3].T + obj_6D[:3, 3]

        hm_obj = self.hm_obj_generator.get_heatmap(obj_kpt2d, bbox_obj_rect)

        # region [normalize data]
        rgb_normalized = normalize_rgb(rgb_patch)
        rgb_tensor = np_to_tensor(rgb_normalized, is_img=True)

        # obj_6D[:3, 3] = obj_6D[:3, 3] - root_joint
        obj_6D[:3, 3] = obj_6D[:3, 3] - root_joint #* to translation raletive to hand wrist
        # obj_verts_cam = obj_verts_ori @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        obj_rot6d = matrix_to_rotation_6d(torch.from_numpy(obj_6D[:3, :3])).numpy()
        obj_pose = np.concatenate([obj_rot6d, obj_6D[:3, 3]], axis=-1)
        root_joint_flip = torch.from_numpy(root_joint)
        obj_CoM = obj_CoM - root_joint
        # endregion

        out = {
            "is_ho3d": True,
            "index": index,
            "rgb_path": rgb_path,
            "rgb": rgb_tensor,
            "root_joint": root_joint,
            "bbox_hand": bbox_hand,
            "bbox_obj": bbox_obj,
            "bbox_hand_rect": bbox_hand_rect,
            "bbox_obj_rect": bbox_obj_rect,
            
            "is_right": True,
            
            "gt_obj": obj_pose,
            
            "root_joint_flip": root_joint_flip,
            
            "obj_name": obj_name,
            "obj_kpt3d": obj_kpt3d,
            "obj_kpt2d": obj_kpt2d,

            "obj_id": obj_id-1,
            "cam_intr": cam_intrinsic,
            "cam_intr_crop": cam_intrinsic_crop,
            "cam_intr_crop_flip": cam_intrinsic_crop,

            "hm_obj": hm_obj,

            "gravity": np.zeros([1, 3]),
            "obj_CoM": np.zeros([1, 3]),
            "is_grasped": self.get_is_grasped(sample_path),
            "force_contact": np.zeros([32]),
            "force_local": np.zeros([32, 3]),
            "force_global": np.zeros([32, 3]),
            "force_point": np.zeros([32, 3]),
        }
        return out
    

class HO3DDatasetForce_Train(HO3DDataset_Force):
    def __init__(
        self,
        data_dir: str,
        is_train: bool,
        aug=BaseDataset._aug,
        cfg=BaseDataset._cfg,
    ):
        super().__init__(data_dir, is_train, aug, cfg)
        self.index_ls = self.load_samples(data_dir, True)

        data_len = len(self.index_ls)
        self.index_ls = self.index_ls[:int(data_len * 1)]

    def __getitem__(self, index):
        return self.get_train_item(index)

class HO3DDatasetForce_Valid(HO3DDataset_Force):
    def __init__(
        self,
        data_dir: str,
        is_train: bool,
        aug=BaseDataset._aug,
        cfg=BaseDataset._cfg,
    ):
        super().__init__(data_dir, is_train, aug, cfg)
        self.index_ls = self.load_samples(data_dir, True)

        self.index_ls = np.array(self.index_ls)[::10]

    def __getitem__(self, index):
        return self.get_train_item(index)

class HO3DDatasetForce_Test(HO3DDataset_Force):
    def __init__(
        self,
        data_dir: str,
        aug=BaseDataset._aug,
        cfg=BaseDataset._cfg,
    ):
        is_train = False
        super().__init__(data_dir, is_train, aug, cfg)
        if self.cfg.clean_data_mode != "2023_NIPS_DeepSimHO":
            self.index_ls = self.load_samples(data_dir)
        else:
            self.index_ls = self.load_HO3Dv2_(data_dir)

    def __getitem__(self, index):
        return self.get_eval_item(index)
    
    def load_samples(self, data_dir):
        evaluation_txt = os.path.join(data_dir, 'evaluation.txt')
        with open(evaluation_txt, 'r') as f:
            eval_ls = f.readlines()

        index_ls = []
        for i in eval_ls:
            i = i.strip()
            i = i.split("/")
            path = os.path.join(data_dir, "evaluation", i[0], "meta", i[1])
            index_ls.append(path)
        return index_ls
    
    def load_HO3Dv2_(self, data_dir):
        # load HO3D v2- testset, align with 2023_NIPS_DeepSimHO
        path = "asset/2023_NIPS_DeepSimHO/cache/HO3D/e154000287d0e832714a8fc8e6a1636b.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        b = data['annot_mapping']

        index_ls = []
        for k, v in b.items():
            for vv in v:
                meta_path = os.path.join(data_dir, "evaluation", k, 'meta', f'{vv["frame_idx"]}')
                index_ls.append(meta_path)
        return index_ls

import os
import json
import time
import pickle
from natsort import natsorted
import numpy as np
import torch
import cv2
import tqdm
from manopth.rodrigues_layer import batch_rodrigues
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d

from lib.dataset.base import (BaseDataset,
                              YCB_CLASSES, 
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
from lib.utils.viz_fn import draw_pts_on_image, depth_to_rgb
from lib.utils.transform_fn import project_pt3d_to_pt2d, rigid_align_AtoB, rigid_transform_3D_AtoB, depth_to_uvd, inverse_project_uvd_to_xyz
from lib.utils.misc_fn import pt2d_to_bbox2d, expand_bbox2d, check_bbox2d, get_rectanglular_bbox2d


class DexYCBDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        is_train: bool,
        aug=BaseDataset._aug,
        cfg=BaseDataset._cfg,
    ):
        self.data_dir = data_dir
        self.clean_data_mode = cfg.clean_data_mode # "2023_CVPR_HFL", "2022_CVPR_ArtiBoost", "stable_grasping"
        super().__init__(is_train, cfg, aug)
        self.index_ls = self.load_samples(data_dir)


    def if_skip_sample(self, sample):
        if self.clean_data_mode == "2023_CVPR_HFL":
            if self.is_train:
                is_skip = not filter_DexYCB_by_HFL(sample)
            else:
                is_skip = False
        elif self.clean_data_mode == "2022_CVPR_ArtiBoost":
            is_skip = not filter_DexYCB_by_ArtiBoost(sample)
        elif self.clean_data_mode == "2023_WACV_DMA":
            if self.is_train:
                return not filter_DexYCB_by_HFL(sample) # 2023_WACV_DMA dose not provide train set index
            else:
                if not hasattr(self, "frame_index_DMA"):
                    cache_path_DMA = "asset/2023_WACV_DMA/test_idx/dex-ycb_test.pkl"
                    with open(cache_path_DMA, "rb") as p_f:
                        dataset_annots = pickle.load(p_f)
                    self.frame_index_DMA = dataset_annots["frame_index"]['img'].tolist()
                    self.frame_index_DMA = [x[10:] for x in self.frame_index_DMA]
                is_skip = sample['color_file'] not in self.frame_index_DMA
        elif self.clean_data_mode == "stable_grasping":
            if not hasattr(self, "__filter_DexYCB_by_steady_grasping"):
                self.__filter_DexYCB_by_steady_grasping = Filter_DexYCB_by_Steady_Grasping()
            is_skip = not self.__filter_DexYCB_by_steady_grasping(sample)
        else:
            raise NotImplementedError
        return is_skip

    def load_samples(self, data_dir: str):
        #* Load sample indices from cache if it exists, otherwise create cache under data_dir
        split = "train" if self.is_train else "test"
        cache_dir = os.path.join(data_dir, "cache", "annotation", split)
        index_path = os.path.join(data_dir, "cache", "annotation", self.clean_data_mode+f"_{split}_index.json")
        index_ls = []
        if not os.path.exists(cache_dir) or not os.path.exists(index_path):
            if self.is_train:
                s0_json = os.path.join(data_dir, "dex_ycb_s0_train_data.json")
            else:
                s0_json = os.path.join(data_dir, "dex_ycb_s0_test_data.json")
            with open(s0_json, "r") as f:
                s0_data = json.load(f)

            os.makedirs(cache_dir, exist_ok=True)
            for k, v in tqdm.tqdm(s0_data.items(), desc="Caching samples"):
                path_i = os.path.join(cache_dir, k+".pkl")
                if not os.path.exists(path_i):
                    with open(path_i, "wb") as f:
                        pickle.dump(v, f)
                if not os.path.exists(index_path) and not self.if_skip_sample(v):
                    index_ls.append(k)
            if not os.path.exists(index_path):
                with open(index_path, "w") as f:
                    json.dump(index_ls, f)
                    
        with open(index_path, "r") as f:
            index_ls = json.load(f)
        self.cache_dir = cache_dir
        return index_ls
    
    """ sample 0: {
        'color_file': '20200820-subject-03/20200820_135508/836212060125/color_000001.jpg', 
        'depth_file': '20200820-subject-03/20200820_135508/836212060125/aligned_depth_to_color_000001.png', 
        'label_file': '20200820-subject-03/20200820_135508/836212060125/labels_000001.npz', 
        'intrinsics': {'fx': 616.640869140625, 'fy': 616.2581787109375, 'ppx': 308.548095703125, 'ppy': 248.52310180664062}, 
        'ycb_ids': [1, 2, 14], 
        'ycb_grasp_ind': 0, 
        'mano_side': 'right', 
        'mano_betas': [-1.1910940408706665,   0.9829702973365784,  0.2026222199201584,  -0.1847328394651413, -0.11520279943943024, 
                        0.39916908740997314, -0.27366748452186584, 0.05005965381860733, -0.23846304416656494, 0.15947425365447998], 
        'is_bop_target': False, 
        'is_grasp_target': False, 
        'joint_3d': [[[-0.27139052748680115, 0.06053471565246582, 0.8559507131576538], 
                      [-0.24978195130825043, 0.07651511579751968, 0.8877922892570496], 
                      [-0.23158833384513855, 0.09853468090295792, 0.9034466743469238], 
                      [-0.2132083773612976,  0.11952574551105499, 0.9081928730010986], 
                      [-0.19884686172008514, 0.15109816193580627, 0.9202889800071716], 
                      [-0.1991189420223236,  0.12172151356935501, 0.8755292892456055], 
                      [-0.19433839619159698, 0.15423040091991425, 0.886191189289093], 
                      [-0.2137507200241089,  0.15948070585727692, 0.8976021409034729], 
                      [-0.22950361669063568, 0.16103442013263702, 0.9153171181678772], 
                      [-0.20501701533794403, 0.13568753004074097, 0.8524799942970276], 
                      [-0.21383874118328094, 0.1635999232530594,  0.8686954379081726], 
                      [-0.2316100299358368,  0.17196223139762878, 0.8829053640365601], 
                      [-0.2523186206817627,  0.17600752413272858, 0.8998862504959106], 
                      [-0.228858083486557,   0.13772159814834595, 0.8351971507072449], 
                      [-0.23062463104724884, 0.1634124517440796,  0.851419985294342], 
                      [-0.24777832627296448, 0.17134378850460052, 0.8692940473556519], 
                      [-0.2648826539516449,  0.17628400027751923, 0.8868674635887146], 
                      [-0.2509855628013611,  0.1387031525373459,  0.8263904452323914], 
                      [-0.2479453831911087,  0.16027770936489105, 0.8293668031692505], 
                      [-0.2544648349285126,  0.17559583485126495, 0.839979887008667], 
                      [-0.26113155484199524, 0.18769574165344238, 0.8533352017402649]]], 
        'joint_2d': [[[113.03392028808594, 292.1062316894531], 
                      [135.05508422851562, 301.63580322265625], 
                      [150.4791717529297,  315.7355041503906], 
                      [163.7847900390625,  329.6278076171875], 
                      [175.31048583984375, 349.70379638671875], 
                      [168.30735778808594, 334.1991271972656], 
                      [173.32107543945312, 355.7750549316406], 
                      [161.70416259765625, 358.0162658691406], 
                      [153.93357849121094, 356.9432067871094], 
                      [160.24916076660156, 346.6116943359375], 
                      [156.7552947998047,  364.58197021484375], 
                      [146.78646850585938, 368.55084228515625], 
                      [135.6484832763672,  369.05621337890625], 
                      [139.578125,         350.14227294921875], 
                      [141.5182342529297,  366.80108642578125], 
                      [132.78451538085938, 369.9917907714844], 
                      [124.37461853027344, 371.0177001953125], 
                      [121.2662124633789,  351.9571838378906], 
                      [124.19873046875,    367.6169128417969], 
                      [121.74192810058594, 377.3504333496094], 
                      [119.84806060791016, 384.0724792480469]]], 
        'pose_y': [[[-0.45545893907546997,  0.8029316067695618,    0.38452279567718506,   -0.03550824150443077], 
                    [-0.8053560256958008,  -0.5556896328926086,    0.2064225971698761,     0.1820649802684784], 
                    [ 0.37941882014274597, -0.2156607061624527,    0.8997399806976318,     0.7921886444091797]], 
                   [[-0.3017919361591339,   0.053226862102746964, -0.9518867135047913,     0.13347534835338593], 
                    [-0.17414282262325287, -0.9847202301025391,    0.00014845197438262403, 0.20003646612167358], 
                    [-0.9373345375061035,   0.16580931842327118,   0.3064497709274292,     1.0176420211791992]], 
                   [[ 0.439127653837204,    0.38661259412765503,  -0.8109855055809021,     0.2152545005083084], 
                    [ 0.373216837644577,   -0.8996020555496216,   -0.22677063941955566,    0.2030183970928192], 
                    [-0.8172369599342346,  -0.2030920535326004,   -0.5393307209014893,     0.7741078734397888]]], 
        'pose_m': [[-0.3741646409034729,    0.8003170490264893,    -2.208287477493286,     #! global rotation
                    -0.20117872953414917,   0.10788445919752121,    1.1836276054382324,    #! mano pose pca mean
                    -0.24507510662078857,   0.301547646522522,      0.43018871545791626, 
                     0.12053897976875305,  -0.15074904263019562,    0.07718818634748459, 
                     0.2517421841621399,    0.42743635177612305,   -0.26533031463623047, 
                    -0.4648362398147583,   -0.36519455909729004,    0.4739420711994171, 
                     0.005123556591570377,  0.4577522277832031,    -0.10931305587291718, 
                    -0.037581589072942734, -0.2533430755138397,     0.3573136031627655, 
                     0.0233029555529356,    0.05453091487288475,    0.15718132257461548, 
                    -0.4632476270198822,   -0.006922731176018715,   0.06739143282175064, 
                     0.35069048404693604,  -0.052652508020401,      0.2296515852212906, 
                    -0.002924699801951647, -0.3929365873336792,     0.12069439142942429, 
                    -0.14226409792900085,  -0.09596936404705048,   -0.1218981221318245, 
                     0.08157256245613098,  -0.07986988872289658,   -0.21278032660484314, 
                     0.22478832304477692,   0.0983148068189621,     0.044425271451473236, 
                    -0.1695689857006073,   -0.07596578449010849,    0.03601725026965141, 
                    -0.3720320463180542,    0.05369240790605545,    0.8496018648147583]],   #! global translation
        'object_seg_file': 'object_render/20200820-subject-03/20200820_135508/836212060125/grasp_object_seg_000001.png'}
    """
    def __getitem__(self, index):
        cache_index = self.index_ls[index] + ".pkl"
        sample = os.path.join(self.cache_dir, cache_index)
        with open(sample, "rb") as f:
            sample = pickle.load(f)

        rgb_path = os.path.join(self.data_dir, sample["color_file"])
        rgb = cv2.imread(rgb_path)[..., ::-1].copy()
        cam_intrinsic = np.array([[sample["intrinsics"]["fx"],  0,                          sample["intrinsics"]["ppx"]],
                                  [0,                           sample["intrinsics"]["fy"], sample["intrinsics"]["ppy"]],
                                  [0,                           0,                          1                          ]], dtype=np.float32)
            
        # hand
        is_right = sample["mano_side"] == "right"
        pose_m =  np.array(sample["pose_m"],dtype=np.float32).squeeze()
        mano_beta = np.array(sample["mano_betas"],dtype=np.float32)
        jt3d = np.array(sample["joint_3d"],dtype=np.float32).squeeze()
        jt2d =  np.array(sample["joint_2d"],dtype=np.float32).squeeze()
        mano_global_rot = pose_m[:3]
        mano_global_transl = pose_m[-3:]
        mano_pose_pca_mean = pose_m[3:-3]
        pca_mat = mano_layer_r.smpl_data["hands_components"] if is_right else mano_layer_l.smpl_data["hands_components"]
        mano_pose_aa_mean = mano_pose_pca_mean @ pca_mat
        mano_hands_mean = mano_layer_r.smpl_data["hands_mean"] if is_right else mano_layer_l.smpl_data["hands_mean"]
        mano_pose_aa_flat = mano_pose_aa_mean + mano_hands_mean
        
        # object
        obj_6D = np.array(sample["pose_y"][sample['ycb_grasp_ind']],dtype=np.float32) # (3, 4)
        obj_id = sample["ycb_ids"][sample['ycb_grasp_ind']]
        obj_name = YCB_CLASSES[obj_id]
        obj_verts_ori = YCB_MESHES[obj_name]["verts"]
        obj_verts_cam = obj_verts_ori @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        obj_kpt3d = YCB_MESHES[obj_name]["kpt3d"] @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        obj_kpt2d = project_pt3d_to_pt2d(obj_kpt3d, cam_intrinsic)
        obj_CoM = YCB_MESHES[obj_name]["CoM"] @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        obj_CoM_z = obj_CoM[2] * 1000 # in mm

         # region [get spatial augmentation data] *checked
        center_jittering, scale_factor, rot_factor = self.get_spatial_aug_params(self.is_train)
        rotmat_3d, rotmat_2d, cam_intrinsic_crop = self.get_augmentation_rotmat(center_jittering, scale_factor, rot_factor, jt2d, obj_kpt2d, cam_intrinsic)

            # region [bbox2d] #! make sure bbox in the image
        rgb_patch = cv2.warpAffine(rgb, rotmat_2d[:2, :], (self.cfg.patch_size, self.cfg.patch_size), flags=cv2.INTER_CUBIC)
        jt2d = jt2d @ rotmat_2d[:2, :2].T + rotmat_2d[:2, 2]
        obj_kpt2d = obj_kpt2d @ rotmat_2d[:2, :2].T + rotmat_2d[:2, 2]
        bbox_hand = pt2d_to_bbox2d(jt2d, mode="x1y1x2y2")
        bbox_hand = expand_bbox2d(bbox_hand, scale_factor=1.15)
        bbox_hand, max_wh_hand = get_rectanglular_bbox2d(bbox_hand)
        is_ok1 = check_bbox2d(bbox_hand, rgb_patch)
        bbox_obj = pt2d_to_bbox2d(obj_kpt2d, mode="x1y1x2y2")
        bbox_obj = expand_bbox2d(bbox_obj, scale_factor=1.05)
        bbox_obj, max_wh_obj = get_rectanglular_bbox2d(bbox_obj)
        is_ok2 = check_bbox2d(bbox_obj, rgb_patch)
        if not is_ok1 or not is_ok2:
            if self.is_train:
                return self.__getitem__(index)
            else:
                cv2.imwrite(f"Tmp_index_{index}.jpg", rgb_patch)
                raise ValueError("bbox out of image")
            # endregion
        # endregion

        obj_depth, obj_depth_back, front_face_map, back_face_map = self.get_obj_front_and_back_depth_map(obj_name, 
                                                                                                         obj_6D, 
                                                                                                         cam_intrinsic, 
                                                                                                         sample["color_file"], 
                                                                                                         background_val=obj_CoM_z)
        obj_depth = obj_depth - obj_CoM_z #* change to relative depth, as predition depth in color image is an ill problem
        obj_depth_back = obj_depth_back - obj_CoM_z

        hand_contact, front_to_joint, front_weight, back_to_joint, back_weight = self.get_hand_and_object_contact(
            mano_pose_aa_flat,
            mano_beta,
            mano_global_rot,
            mano_global_transl,
            is_right,
            obj_name,
            obj_6D,
            front_face_map,
            back_face_map,
            sample["color_file"],
        )
        hand_contact = np.clip(hand_contact, 0, 1)
        
       
        # region [do 3D spatial augmentation]  #* checked
        # rot hand 3d. #* checked
        jt3d = jt3d @ rotmat_3d.T
        # jt2d_proj = project_pt3d_to_pt2d(jt3d, cam_intrinsic) # indentical to jt2d
        mano_global_rotmat = axis_angle_to_matrix(torch.tensor(mano_global_rot)).numpy()
        mano_global_rotmat = rotmat_3d @ mano_global_rotmat
        mano_global_rot = matrix_to_axis_angle(torch.tensor(mano_global_rotmat)).numpy()
        gt_hand_vert, _jt3d = get_hand_vert(mano_pose_aa_flat, mano_beta, mano_global_rot, mano_global_transl, is_right)
        mano_global_transl = mano_global_transl + (jt3d[0] - _jt3d[0]) #! correct translation
        gt_hand_vert = gt_hand_vert + (jt3d[0] - _jt3d[0])

        # rot object. #* checked
        obj_6D[:3, :3] = rotmat_3d @ obj_6D[:3, :3]
        obj_6D[:3, 3] = rotmat_3d @ obj_6D[:3, 3]
        obj_kpt3d = YCB_MESHES[obj_name]["kpt3d"] @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        # obj_kpt2d_proj = project_pt3d_to_pt2d(obj_kpt3d, cam_intrinsic) # indentical to obj_kpt2d
        # obj_verts_cam = obj_verts_ori @ obj_6D[:3, :3].T + obj_6D[:3, 3]
        # endregion

        # region [crop image]
        # crop image. #* checked
        # obj_depth_front_patch = cv2.warpAffine(obj_depth, rotmat_2d[:2, :], (self.cfg.patch_size, self.cfg.patch_size), flags=cv2.INTER_NEAREST)
        # obj_depth_back_patch = cv2.warpAffine(obj_depth_back, rotmat_2d[:2, :], (self.cfg.patch_size, self.cfg.patch_size), flags=cv2.INTER_NEAREST)
        # front_to_joint_patch = cv2.warpAffine(front_to_joint, rotmat_2d[:2, :], (self.cfg.patch_size, self.cfg.patch_size), flags=cv2.INTER_NEAREST)
        # back_to_joint_patch = cv2.warpAffine(back_to_joint, rotmat_2d[:2, :], (self.cfg.patch_size, self.cfg.patch_size), flags=cv2.INTER_NEAREST)
        # front_weight_patch = cv2.warpAffine(front_weight, rotmat_2d[:2, :], (self.cfg.patch_size, self.cfg.patch_size), flags=cv2.INTER_NEAREST)
        # back_weight_patch = cv2.warpAffine(back_weight, rotmat_2d[:2, :], (self.cfg.patch_size, self.cfg.patch_size), flags=cv2.INTER_NEAREST)
        # endregion

        # region [color augmentation]  #* checked
        if self.is_train:
            rgb_patch = self.image_augmentor.run_color(rgb_patch)



        # region [flip left hand]
        if not is_right:
            rgb_patch = rgb_patch[:, ::-1].copy()
            jt2d[:, 0] = rgb_patch.shape[1]  - jt2d[:, 0] - 1
            # jt3d[:, 0] = -jt3d[:, 0]
            obj_kpt2d[:, 0] = rgb_patch.shape[1] - obj_kpt2d[:, 0] - 1
            bbox_hand[[0, 2]] = rgb_patch.shape[1] - bbox_hand[[2, 0]] - 1
            bbox_obj[[0, 2]] = rgb_patch.shape[1] - bbox_obj[[2, 0]] - 1

            mano_pose_aa_mean = mano_pose_aa_mean.reshape(-1, 3)
            mano_pose_aa_mean[:, 1:] *= -1
            mano_pose_aa_mean = mano_pose_aa_mean.reshape(-1)
            mano_global_rot[1:] *= -1
            mano_global_transl[0] *= -1
            cam_intrinsic_flip = cam_intrinsic.copy()
            cam_intrinsic_flip[0, 2] = rgb.shape[1] - cam_intrinsic_flip[0, 2] - 1
            mano_pose_aa_flat = mano_pose_aa_mean + mano_layer_r.smpl_data["hands_mean"]

            #! correct translation  # TODO: correct translation by pa-align
            _, _jt3d = get_hand_vert(mano_pose_aa_flat, mano_beta, mano_global_rot, mano_global_transl, True)
            mano_global_transl = mano_global_transl + (jt3d.mean(0) - _jt3d.mean(0))

        hand_vert, _jt3d = get_hand_vert(mano_pose_aa_flat, mano_beta, mano_global_rot, mano_global_transl, True)
        # endregion

        # region [Heatmap]
        #* checked
        jt2d_hm = jt2d - bbox_hand[:2]
        jt2d_hm = jt2d_hm / (max_wh_hand) * (self.cfg.heatmap_size - 1)
        hm_hand = self.hm_hand_generator(jt2d_hm)

        obj_kpt2d_hm = obj_kpt2d - bbox_obj[:2]
        obj_kpt2d_hm = obj_kpt2d_hm / (max_wh_obj) * (self.cfg.heatmap_size - 1)
        hm_obj = self.hm_hand_generator(obj_kpt2d_hm)

        # TODO: generate hintmap for object center and wrist
        # obj_ct_uv_hm = obj_kpt2d_hm[14]
        # obj_ct_d = obj_kpt3d[14, 2] - jt3d[0, 2]
        # wrist_uv_hm = jt2d_hm[0]
        # endregion

        # region [normalization] #TODO: to be checked
        rgb_normalized = normalize_rgb(rgb_patch)
        rgb_tensor = np_to_tensor(rgb_normalized, is_img=True)

        if self.is_train:
            rgb_tensor = self.image_augmentor.run_random_erasing(rgb_tensor) #* checked, random erasing augmentation

        # front_depth_normalized = normalize_depth(obj_depth_front_patch)
        # front_depth_tensor = np_to_tensor(front_depth_normalized)
        # back_depth_normalized = normalize_depth(obj_depth_back_patch)
        # back_depth_tensor = np_to_tensor(back_depth_normalized)
        # depth_tensor = torch.stack([front_depth_tensor, back_depth_tensor], dim=0) # (2, H, W)

        # front_contact_normalized = normalize_contact(front_to_joint_patch, front_weight_patch)
        # front_contact_tensor = np_to_tensor(front_contact_normalized)
        # back_contact_normalized = normalize_contact(back_to_joint_patch, back_weight_patch)
        # back_contact_tensor = np_to_tensor(back_contact_normalized)
        # contact_tensor = torch.stack([front_contact_tensor, back_contact_tensor], dim=0) # (2, 7, H, W)

        #! obj pose is never flipped while hand pose is flipped if left hand presented
        root_joint = jt3d[0]
        obj_6D[:3, 3] = obj_6D[:3, 3] - root_joint #* to translation raletive to hand wrist
        obj_6D = torch.from_numpy(obj_6D).to(torch.float32)
        obj_rot6d = matrix_to_rotation_6d(obj_6D[:3, :3])
        obj_pose = torch.cat([obj_rot6d, obj_6D[:3, 3]], dim=-1)

        root_joint = torch.tensor(root_joint, dtype=torch.float32)
        mano_params = np.concatenate([mano_global_rot, mano_pose_aa_flat, mano_beta]) # (58,) #* checked, use mean pose instead of flat pose
        mano_params = torch.tensor(mano_params, dtype=torch.float32)
        jt3d = torch.tensor(jt3d, dtype=torch.float32)
        cam_intrinsic = torch.tensor(cam_intrinsic, dtype=torch.float32)
        cam_intrinsic_crop = torch.tensor(cam_intrinsic_crop, dtype=torch.float32)
        # endregion

        out = {
            "rgb_path": rgb_path,
            "rgb": rgb_tensor,
            "root_joint": root_joint,
            "bbox_hand": bbox_hand,
            "bbox_obj": bbox_obj,
            "hm_hand": hm_hand,
            "hm_obj": hm_obj,
            "is_right": is_right,
            "gt_jt2d": jt2d,
            "gt_obj2d": obj_kpt2d,

            "gt_obj": obj_pose,
            "gt_mano": mano_params, # flipped
            "gt_joint": jt3d, # never flipped
            "gt_hand_vert": gt_hand_vert, # never flipped
            "obj_name": obj_name,
            "cam_intr": cam_intrinsic,
            "cam_intr_crop": cam_intrinsic_crop,
            "gt_hand_contact": hand_contact,

            # "gt_mano": torch.tensor(mano_params, dtype=torch.float32),
            # "gt_hand_transl": torch.tensor(mano_global_transl, dtype=torch.float32),
            # 'gt_jt3d': torch.tensor(jt3d, dtype=torch.float32),
            # 'gt_hand_vert': torch.tensor(hand_vert, dtype=torch.float32),
            # "depth": depth_tensor,
            # "contact": contact_tensor,
            # "bbox_mask_unite": bbox_mask_unite,
            # "bbox_mask_inter": bbox_mask_inter,
            # "cam_intrinsic": cam_intrinsic,
            # "obj_id": obj_id,
            # "obj_name": obj_name,
            # "hand_contact": hand_contact,
        }
        return out


import time
import torch
import torch.nn.functional as F
import copy
from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    quaternion_to_matrix,
    matrix_to_quaternion,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)

from lib.utils.hand_fn import MANO_PARAMS_LEVEL, MANO_JOINT_LEVEL
from lib.utils.transform_fn import average_quaternion
from lib.model.head_object import HeadObject
from lib.model.head_mano import HeadMano
from lib.configs.args import cfg
from lib.model.physics import from_local_to_global

def project_point_by_cam_intrinsic(**kwargs):
    ''' project hand joint to 2D image plane
        pt3d_cam: (bs, ..., 3)
        cam_intrinsic: (bs, 3, 3)
    '''
    shape = kwargs['pt3d_cam'].shape
    joint2d = torch.einsum('b...ij,blj->b...il', kwargs['pt3d_cam'], kwargs['cam_intrinsic'])
    joint2d = joint2d[..., :2] / joint2d[..., 2:]
    return joint2d

# TODO: to be checked
def inverse_project_point_by_cam_intrinsic(uvd, cam_intr):
    """inverse project uvd to xyz
    Args:
        uvd: [B, ..., 3]
        cam_intr: [B, 3, 3]
    Returns:
        xyz: [B, ..., 3]
    """
    cam_intr_inv = torch.inverse(cam_intr)
    xyz = torch.ones_like(uvd)
    xyz[..., :2] = uvd[..., :2]
    xyz = xyz * uvd[..., 2:]
    xyz = torch.einsum('b...i,bki->b...k', xyz, cam_intr_inv)
    return xyz

def average_rot6d(rot6d, weights=None):
    if weights is None:
        weights = torch.ones_like(rot6d[..., 0]) / rot6d.shape[-2]
    quat = matrix_to_quaternion(rotation_6d_to_matrix(rot6d))
    quat_mean = average_quaternion(quat, weights)
    rot6d_mean = matrix_to_rotation_6d(quaternion_to_matrix(quat_mean))
    return rot6d_mean

#* checked
class HandSelector:
    def __init__(self, mano_fn:HeadMano.get_hand_verts):
        self.mano_fn = mano_fn
            
    def select_topk_hand(self, **kwargs):
        ''' mode: 'heatmap' or 'heatmap_cascade'
            if mode == 'heatmap':
                pose: (bs*sample_num, 48)
                shape: (bs*sample_num, 10)
                root_joint: (bs, 3), 
                cam_intrinsic: (bs, 3, 3), 
                heatmap: (bs, 21, H, W)
                bbox: (bs, 4)
                k: (k,)
                fuse_index: (n,)
                observe_index: (m,)
                is_independent: bool
            
            if mode == 'heatmap_cascade':
                pose: (bs*sample_num, 48)
                pose_regression: (bs*sample_num, 48)
                shape: (bs*sample_num, 10)
                root_joint: (bs, 3), 
                cam_intrinsic: (bs, 3, 3), 
                heatmap: (bs, 21, H, W)
                bbox: (bs, 4)
                k: (k,)
                fuse_index: (n,)
                observe_index: (m,)
                is_independent: bool
        '''
        bs = kwargs['root_joint'].shape[0]
        if kwargs['mode'] == 'heatmap': # checked
            
            fused_data = self.select_topk_hand_by_observed_heatmap_and_fuse_by_index(
                pose=kwargs['pose'],
                shape=kwargs['shape'],
                root_joint=kwargs['root_joint'],
                cam_intrinsic=kwargs['cam_intrinsic'],
                heatmap=kwargs['heatmap'],
                bbox=kwargs['bbox'],
                k=kwargs['k'],
                fuse_index=list(range(48)),
                observe_index=list(range(21)),
                is_independent=False,
            )
            fused_pose = fused_data['fused_pose'][:, 0] # (bs, 58)
            shape = kwargs['shape'].reshape(bs, -1, 10)[:, 0]
            fused_mano = torch.cat((fused_pose, shape), dim=-1)
            fused_vert, fused_joint = self.mano_fn(pose=fused_pose, shape=shape)
            fused_vert, fused_joint = fused_vert.reshape(bs, 778, 3), fused_joint.reshape(bs, 21, 3)

            return {
                'topk': fused_data['topk'],
                'diff_topk_vert': fused_data['topk_vert'],
                'diff_topk_joint': fused_data['topk_joint'],
                'diff_topk_fused_mano': fused_mano,
                'diff_topk_fused_vert': fused_vert,
                'diff_topk_fused_joint': fused_joint,
                'fused_data_ls': [fused_data]
            }

        if kwargs['mode'] == 'heatmap_cascade':

            pose=kwargs['pose'].clone()
            shape=kwargs['shape'].clone()

            extra_pose = torch.zeros_like(pose).reshape(bs, -1, 48)
            extra_pose = extra_pose + kwargs['pose_regression'][:, None].clone()
            pose = pose.reshape(bs, -1, 48)
            num_candidate = pose.shape[1]
            pose = torch.cat((pose, extra_pose), dim=1).reshape(-1, 48) #* useful, MJE: 11.22 -> 11.15, regression result as candidate
            shape = shape.reshape(bs, -1, 10).repeat(1, 2, 1).reshape(-1, 10)

            fused_data_ls = []
            for level_i in range(4):
                fuse_idx = MANO_PARAMS_LEVEL[level_i]
                # observe_idx = MANO_JOINT_LEVEL[level_i+1] # deprecated, MJE: 11.48 -> 11.61, only the direct children which is not very useful. This may be caused by the occlusion of the direct children joints.
                observe_idx = []
                for j in range(level_i+1, 5):  #* useful MJE: 11.48 -> 11.22 
                    observe_idx.extend(MANO_JOINT_LEVEL[j])

                if level_i in [0]: # only useful for the wrist
                    pose = pose.view(bs, -1, 48)
                    pose[:, num_candidate:, fuse_idx] = pose[:, :num_candidate, fuse_idx]
                    pose = pose.reshape(-1, 48)

                fused_data_i = self.select_topk_hand_by_observed_heatmap_and_fuse_by_index(
                    pose=pose,
                    shape=shape,
                    root_joint=kwargs['root_joint'],
                    cam_intrinsic=kwargs['cam_intrinsic'],
                    heatmap=kwargs['heatmap'],
                    bbox=kwargs['bbox'],
                    k=kwargs['k'],
                    fuse_index=fuse_idx,
                    observe_index=observe_idx,
                    is_independent=False if level_i == 0 else True,
                )

                pose = fused_data_i['fused_pose'].reshape(-1, 48)
                fused_data_ls.append(fused_data_i)

            fused_pose = fused_data_ls[-1]['fused_pose'][:, 0] # (bs, 58)
            shape = kwargs['shape'].reshape(bs, -1, 10)[:, 0]
            fused_mano = torch.cat((fused_pose, shape), dim=-1)
            fused_vert, fused_joint = self.mano_fn(pose=fused_pose, shape=shape)
            fused_vert, fused_joint = fused_vert.reshape(bs, 778, 3), fused_joint.reshape(bs, 21, 3)

            return {
                'topk': fused_data_ls[-1]['topk'],
                'diff_topk_vert': fused_data_ls[-1]['topk_vert'],
                'diff_topk_joint': fused_data_ls[-1]['topk_joint'],
                'diff_topk_fused_mano': fused_mano,
                'diff_topk_fused_vert': fused_vert,
                'diff_topk_fused_joint': fused_joint,
                'diff_vert': fused_data_ls[0]['vert'],
                'diff_joint': fused_data_ls[0]['joint'],
                # 'fused_data_ls': fused_data_ls
            }
        else:
            raise NotImplementedError


    def select_topk_hand_by_observed_heatmap_and_fuse_by_index(self, **kwargs):
        """
            pose: (bs*sample_num, 48)
            shape: (bs*sample_num, 10)
            root_joint: (bs, 3)
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            fuse_index: (n, )
            observe_index: (m, )
            is_independent: bool
        """
        bs, J, H, W = kwargs['heatmap'].shape
        K = kwargs['k']
        vert, joint = self.mano_fn(pose=kwargs['pose'], shape=kwargs['shape'])
        vert, joint = vert.reshape(bs, -1, 778, 3), joint.reshape(bs, -1, 21, 3)
        joint_cam = joint + kwargs['root_joint'][:, None, None]
        pose = kwargs['pose'].reshape(bs, -1, 16, 3)

        pt2d = project_point_by_cam_intrinsic(pt3d_cam=joint_cam, cam_intrinsic=kwargs['cam_intrinsic'])
        bbox = kwargs['bbox'][:, None, None, :]
        pt2d = pt2d - bbox[..., :2]
        pt2d = 2 * pt2d / (bbox[..., 2:] - bbox[..., :2]) - 1 # (-1, 1)

        heat_val = []
        for i in kwargs['observe_index']: # TODO: to be checked
            grid_i = pt2d[:, :, [i]]
            heatmap_i = kwargs['heatmap'][:, [i]]
            heat_val_i = F.grid_sample(heatmap_i, grid_i, align_corners=False, mode='bicubic')
            heat_val_i = heat_val_i.squeeze(1)
            heat_val.append(heat_val_i)
        heat_val = torch.concat(heat_val, dim=-1) # (bs, sample_num, m)

        if not kwargs['is_independent']: #* checked
            heat_val = heat_val.sum(dim=-1)
            val, topk = heat_val.topk(K, dim=1) # (bs, K), (bs, K)
            weight = (val+1e-8) / (val.sum(dim=1, keepdim=True)+1e-8) # (bs, K)

            idx_tensor = torch.arange(bs, device=heat_val.device)[:, None].repeat(1, K)
            topk_pose = kwargs['pose'].reshape(bs, -1, 48)[idx_tensor, topk]
            topk_idx_pose = topk_pose[:, :, kwargs['fuse_index']]
            topk_idx_pose_aa = topk_idx_pose.reshape(bs, K, -1, 3) # (bs, k, n, 3)
            topk_idx_pose_quat = axis_angle_to_quaternion(topk_idx_pose_aa) # (bs, k, n, 4)
            topk_idx_pose_quat = topk_idx_pose_quat.permute(0, 2, 1, 3)
            fuse_idx_pose_quat = average_quaternion(topk_idx_pose_quat, weight[:, None]) # (bs, n, 4) # TODO weighted average
            fused_idx_pose_aa = quaternion_to_axis_angle(fuse_idx_pose_quat) # (bs, n, 3)
            fused_idx_pose_aa = fused_idx_pose_aa.reshape(bs, -1)

            fused_pose = kwargs['pose'].reshape(bs, -1, 48)
            fused_pose[:, :, kwargs['fuse_index']] = fused_pose[:, :, kwargs['fuse_index']] * 0 + fused_idx_pose_aa[:, None]

            topk_vert = vert[idx_tensor, topk]
            topk_joint = joint[idx_tensor, topk]

        else:
            M, N = len(kwargs['observe_index']), len(kwargs['fuse_index'])
            assert M % (N // 3) == 0
            n_observed = M // (N // 3)
            heat_val = heat_val.reshape(bs, -1, n_observed, N//3).mean(dim=-2) # (bs, K, n)
            val, topk = heat_val.topk(K, dim=1) # (bs, K, n), (bs, K, n)
            weight = (val+1e-8) / (val.sum(dim=1, keepdim=True)+1e-8) # (bs, K, n)
            weight = weight.permute(0, 2, 1) # (bs, n, K)

            pose = kwargs['pose'].reshape(bs, -1, 16, 3) # (bs, sample_num, 48)
            idx_tensor1 = torch.arange(bs, device=heat_val.device)[:, None, None].repeat(1, K, N//3)
            idx_tensor2 = torch.tensor(kwargs['fuse_index'], dtype=torch.long, device=heat_val.device)[None, None].repeat(bs, K, 1).reshape(bs, K, -1, 3)
            idx_tensor2 = idx_tensor2[:, :, :, 0] // 3  # TODO: To be more elegant

            topk_pose = pose[idx_tensor1, topk, idx_tensor2] # (bs, K, M, 3)
            topk_idx_posed_quat = axis_angle_to_quaternion(topk_pose)
            topk_idx_posed_quat = topk_idx_posed_quat.permute(0, 2, 1, 3) # (bs, M, K, 4)
            fused_idx_pose_quat = average_quaternion(topk_idx_posed_quat, weight) # (bs, M, 4)
            fused_idx_pose_aa = quaternion_to_axis_angle(fused_idx_pose_quat) # (bs, M, 3)
            fused_idx_pose_aa = fused_idx_pose_aa.reshape(bs, -1)

            fused_pose = kwargs['pose'].reshape(bs, -1, 48)
            fused_pose[:, :, kwargs['fuse_index']] = fused_pose[:, :, kwargs['fuse_index']] * 0 + fused_idx_pose_aa[:, None]
            
            topk_vert = vert[idx_tensor1, topk]
            topk_joint = joint[idx_tensor1, topk]

        return {
            'val': val,
            'topk': topk,
            'fused_idx_pose': fused_idx_pose_aa,
            'fused_pose': fused_pose,
            'topk_vert': topk_vert,
            'topk_joint': topk_joint,
            'vert': vert,
            'joint': joint,
        }
    
    # it takes about 0.03s for batch size 16
    def select_topk_hand_by_physics(self, **kwargs):
        ''' mode: 'physics'
            pose_dif: (bs, n, 48)
            pose_reg: (bs, 48)
            pose_fuse: (bs, 48)

            shape: (bs*sample_num, 10)
            root_joint: (bs, 3), 
            cam_intrinsic: (bs, 3, 3), 
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: (k,)
            fuse_index: (n,)
            observe_index: (m,)
            is_independent: bool
            obj_vert: (bs, 2048, 3)
            is_grasped: (bs, )
        '''
        bs = kwargs['root_joint'].shape[0]

        pose_d=kwargs['pose_dif']
        pose_r = kwargs['pose_reg'][:, None]
        pose_f = kwargs['pose_fuse'][:, None]
        pose_src = torch.cat((pose_d, pose_r, pose_f), dim=1) # (bs, n+2, 48)
        shape=kwargs['shape'][:, None].repeat(1, pose_src.size(1), 1) # (bs, n+2, 10)

        # region [get new candidates]
        candidate_dt = {}
        for level_i in MANO_PARAMS_LEVEL:
            if level_i >= 2: # Proximal and Distal joint
                fuse_idx = MANO_PARAMS_LEVEL[level_i]
                pose = pose_src.clone()
                pose[:, : , fuse_idx] = pose[:, : , fuse_idx] * 0 + kwargs['pose_fuse'][:, None, fuse_idx]

                observe_idx = []
                for j in range(level_i+1, 5):  #* useful MJE: 11.48 -> 11.22 
                    observe_idx.extend(MANO_JOINT_LEVEL[j])

                fused_data = self.select_topk_hand_by_observed_heatmap_and_fuse_by_index(
                    pose=pose.reshape(-1, 48).clone(),
                    shape=shape.reshape(-1, 10),
                    root_joint=kwargs['root_joint'],
                    cam_intrinsic=kwargs['cam_intrinsic'],
                    heatmap=kwargs['heatmap'],
                    bbox=kwargs['bbox'],
                    k=kwargs['k'],
                    fuse_index=fuse_idx,
                    observe_index=observe_idx,
                    is_independent=False if level_i == 0 else True,
                )
                candidate_dt[level_i] = fused_data['topk']
        
        idx_tensor1 = torch.arange(bs, device=pose_src.device)[:, None, None].repeat(1, kwargs['k'], len(MANO_PARAMS_LEVEL[2])//3)
        idx_tensor2 = torch.tensor(MANO_PARAMS_LEVEL[2], dtype=torch.long, device=pose_src.device).reshape(-1, 3)[:, 0] // 3
        idx_tensor2 = idx_tensor2[None, None].repeat(bs, kwargs['k'], 1)
        pose1 = pose_src.reshape(bs, -1, 16, 3)[idx_tensor1, candidate_dt[2], idx_tensor2] # (bs, k, 15)
        pose1 = pose1.reshape(bs, -1, len(MANO_PARAMS_LEVEL[2]))

        idx_tensor3 = torch.tensor(MANO_PARAMS_LEVEL[3], dtype=torch.long, device=pose_src.device).reshape(-1, 3)[:, 0] // 3
        idx_tensor3 = idx_tensor3[None, None].repeat(bs, kwargs['k'], 1)
        pose2 = pose_src.reshape(bs, -1, 16, 3)[idx_tensor1, candidate_dt[3],idx_tensor3] # (bs, k, 15)
        pose2 = pose2.reshape(bs, -1, len(MANO_PARAMS_LEVEL[3]))

        pose1 = pose1[:, :, None].repeat(1, 1, kwargs['k'], 1) # (bs, k, k, 15)
        pose2 = pose2[:, None].repeat(1, kwargs['k'], 1, 1)
        pose_c = torch.cat((pose1, pose2), dim=-1) # (bs, k, k, 30)
        pose_c = pose_c.reshape(bs, -1, 30) # (bs, k*k, 30)

        pose_candi = pose_f.repeat(1, kwargs['k']*kwargs['k'], 1) # (bs, k*k, 48)
        update_mano_params_idx = MANO_PARAMS_LEVEL[2] + MANO_PARAMS_LEVEL[3]
        pose_candi[:, :, update_mano_params_idx] = pose_c

        shape = kwargs['shape'][:, None].repeat(1, pose_candi.size(1), 1).reshape(-1, 10)
        pose_candi = pose_candi.reshape(-1, 48)
        vert_candi, joint_candi = self.mano_fn(pose=pose_candi, shape=shape)
        vert_candi, joint_candi = vert_candi.reshape(bs, -1, 778, 3), joint_candi.reshape(bs, -1, 21, 3)
        vert_candi_cam = vert_candi + kwargs['root_joint'][:, None, None]
        # endregion

        # region [force-based selection]
        force_local = kwargs['force_local'].reshape(bs, -1, 32, 3).repeat(1, vert_candi_cam.size(1), 1, 1).reshape(-1, 32, 3)
        vert_candi_cam = vert_candi_cam.reshape(-1, 778, 3)
        force_point, force_global = from_local_to_global(force_local, vert_candi_cam) # (-1, 32, 3), (-1, 32, 3)
        
        force_local_norm = kwargs['force_local'].norm(dim=-1) # (bs, 32)
        force_weight = force_local_norm / force_local_norm.sum(dim=-1, keepdim=True) # (bs, 32)

        force_point = force_point.reshape(bs, -1, 32, 3) # (bs, c, 32, 3)
        obj_vert = kwargs['obj_vert'][:, None].repeat(1, force_point.size(1), 1, 1) # (bs, c, 2048, 3)
        cdist = cdist_memory_save(force_point, obj_vert) # (bs, c, 32)

        score = -(cdist * force_weight[:, None]).sum(-1)
        val, topk = score.topk(kwargs['k'], dim=1)
        # endregion 

        # region [fuse topk candidates]
        pose_candi = pose_candi.reshape(bs, -1, 16, 3) # (bs, sample_num, 16, 3)
        idx_tensor1 = torch.arange(bs, device=pose_candi.device)[:, None].repeat(1, kwargs['k'])

        topk_pose = pose_candi[idx_tensor1, topk] # (bs, K, M, 3)
        topk_idx_posed_quat = axis_angle_to_quaternion(topk_pose)
        topk_idx_posed_quat = topk_idx_posed_quat.permute(0, 2, 1, 3)
        fused_idx_pose_quat = average_quaternion(topk_idx_posed_quat)
        fused_idx_pose_aa = quaternion_to_axis_angle(fused_idx_pose_quat) # (bs, M, 3)
        fused_idx_pose_aa = fused_idx_pose_aa.reshape(bs, -1)
        fused_pose = fused_idx_pose_aa
        
        shape = kwargs['shape']
        fused_mano = torch.cat((fused_pose, shape), dim=-1)
        fused_vert, fused_joint = self.mano_fn(pose=fused_pose, shape=shape)
        # endregion
        

        # from lib.utils.viz_fn import get_random_color
        # save_dt = {}
        # # for i in range(vert_candi_cam.size(1)):
        # #     save_dt[f"candi{i}_{get_random_color(True)}"] = vert_candi_cam[0, i].detach().cpu().numpy()

        # ref_vert, ref_joint = self.mano_fn(pose=kwargs['pose_fuse'], shape=kwargs['shape'])
        # save_dt['ref_#FF0000'] = (ref_vert[0] + kwargs['root_joint'][0]).detach().cpu().numpy()
        # save_dt['obj_#000000'] = kwargs['obj_vert'][0].detach().cpu().numpy()
        # save_dt['sel_#FF00FF'] = (fused_vert[0] + kwargs['root_joint'][0]).detach().cpu().numpy()

        # with open("tmp.pkl", "wb") as f:
        #     import pickle
        #     pickle.dump(save_dt, f)
        # print(vert_candi_cam.shape)
        # exit()

        return {
            'fused_mano': fused_mano,
            'fused_vert': fused_vert,
            'fused_joint': fused_joint,
        }



class ObjectSelector:
    def __init__(self, obj_fn:HeadObject):
        self.obj_layer = obj_fn

    def select_topk_object(self, **kwargs):
        ori_pose6d = kwargs['pose6d'].clone()
        
        topk, weight = self.select_topk_object_by_heatmap(**kwargs)
        pose6d_fused = self.fuse_topk(topk=topk, weight=weight, **kwargs) # (bs, 9)
        fused_trans1 = pose6d_fused[:, 6:]

        kwargs['pose6d'][..., 6:] = ori_pose6d[..., 6:] * 0 + fused_trans1[:, None]
        topk, weight = self.select_topk_object_by_heatmap(**kwargs)
        pose6d_fused = self.fuse_topk(topk=topk, weight=weight, **kwargs)
        fused_rot1 = pose6d_fused[:, :6]

        kwargs['pose6d'] = ori_pose6d.clone()
        kwargs['pose6d'][..., :6] = ori_pose6d[..., :6] * 0 + fused_rot1[:, None]
        topk_trans2, weight_trans2 = self.select_topk_object_by_heatmap(**kwargs)
        trans2_candidate = self.topk_select(topk_trans2, kwargs['pose6d']) # (bs, k, 9)
        trans2_candidate[:, :, :6] = trans2_candidate[:, :, :6] * 0

        kwargs['pose6d'] = ori_pose6d.clone()
        kwargs['pose6d'][..., 6:] = ori_pose6d[..., 6:] * 0 + fused_trans1[:, None]
        topk_rot2, weight_rot2 = self.select_topk_object_by_heatmap(**kwargs)
        rot2_candidate = self.topk_select(topk_rot2, kwargs['pose6d'])
        rot2_candidate[:, :, 6:] = rot2_candidate[:, :, 6:] * 0

        if use_force_selestion:=True: #* useful, HO3Dv2-ADDS: 15.35 -> 14.15
            new_candidate = trans2_candidate[:, None] + rot2_candidate[:, :, None] # (bs, k, k, 9)
            new_candidate = new_candidate.reshape(ori_pose6d.size(0), -1, 9) # (bs, k*k, 9)
            kwargs['pose6d'] = new_candidate

            topk_physics, weight_physics = self.select_topk_object_by_physics2(**kwargs)        
            topk_heatmap, weight_heatmap = self.select_topk_object_by_heatmap(**kwargs)

            ungrasp_idx = torch.arange(ori_pose6d.size(0), device=kwargs['pose6d'].device)
            ungrasp_idx = ungrasp_idx[~kwargs['is_grasped']]
            new_topk = topk_physics.clone()
            new_topk[ungrasp_idx] = topk_heatmap[ungrasp_idx]
            new_weight = weight_physics.clone()
            new_weight[ungrasp_idx] = weight_heatmap[ungrasp_idx]
            pose6d_fused = self.fuse_topk(topk=new_topk, **kwargs)
        else:
            trans2 = self.fuse_topk(topk=topk_trans2, **kwargs)[:, 6:]
            rot2 = self.fuse_topk(topk=topk_rot2, **kwargs)[:, :6]
            pose6d_fused = torch.cat([rot2, trans2], dim=-1)

        pose6d_fused = pose6d_fused.float()
        pose6d = pose6d_fused.clone()
        pose6d[..., 6:] = pose6d[..., 6:] + kwargs['root_joint']
        obj_vert_fused = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts')
        obj_vert_fused = self.obj_layer.flip_pt3d(obj_vert_fused, kwargs['is_right']) 

        return {
            # 'topk': new_topk,
            'pose6d_fused': pose6d_fused,
            'pose6d_candidate': kwargs['pose6d'],
            'obj_vert_fused': obj_vert_fused,
        }
    
    def topk_select(self, topk, pose6d):
        bs = pose6d.size(0)
        idx_tensor = torch.arange(bs, device=pose6d.device)[:, None].repeat(1, topk.size(1))
        return pose6d[idx_tensor, topk]
    
    def fuse_topk(self, **kwargs):
        # bs = kwargs['pose6d'].size(0)
        # idx_tensor = torch.arange(bs, device=kwargs['pose6d'].device)[:, None].repeat(1, kwargs['k'])
        # pose6d_topk = kwargs['pose6d'][idx_tensor, kwargs['topk']]
        pose6d_topk = self.topk_select(kwargs['topk'], kwargs['pose6d'])
        if kwargs.get('weight', None) is None:
            trans_fused = pose6d_topk[:, :, 6:].mean(dim=1)
        else:
            weight = kwargs['weight']
            trans_fused = (pose6d_topk[:, :, 6:] * weight[:, :, None]).sum(dim=1)

        rot6d_topk = pose6d_topk[..., :6]
        rot6d_fused = average_rot6d(rot6d_topk, weights=kwargs.get('weight', None))
        pose6d_fused = torch.cat([rot6d_fused, trans_fused], dim=-1)
        return pose6d_fused # (bs, 9)

    def select_topk_object_by_heatmap(self, **kwargs):
        """ pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            is_right: (bs, )
            observe_index: (m,)
        """
        pose6d = kwargs['pose6d'].clone().float()
        root_joint = kwargs["root_joint"].unsqueeze(1)
        pose6d[..., 6:] =  pose6d[..., 6:] + root_joint
        obj_pt3d = self.obj_layer(pose6d, kwargs['obj_name'])
        obj_pt3d = self.obj_layer.flip_pt3d(obj_pt3d, kwargs['is_right'])
        pt2d = project_point_by_cam_intrinsic(pt3d_cam=obj_pt3d, cam_intrinsic=kwargs['cam_intrinsic'])

        bbox = kwargs['bbox'][:, None, None, :]
        pt2d = pt2d - bbox[..., :2]
        pt2d = 2 * pt2d / (bbox[..., 2:] - bbox[..., :2]) - 1 # to (-1, 1)

        bs, J, H, W = kwargs['heatmap'].shape
        observe_index = kwargs.get('observe_index', list(range(J)))
        assert len(observe_index) == J

        heatval = []
        for i in observe_index:
            grid_i = pt2d[:, :, [i]]
            heatmap_i = kwargs['heatmap'][:, [i]]
            heatval_i = F.grid_sample(heatmap_i, grid_i, align_corners=False, mode='bicubic')
            heatval_i = heatval_i.squeeze(1)
            heatval.append(heatval_i)
        heatval = torch.concat(heatval, dim=-1) # (bs, sample_num, 21)
        heatval = heatval.sum(dim=-1) # (bs, sample_num)

        val, topk = heatval.topk(kwargs['k'], dim=1)
        weight = (val+1e-8) / (val.sum(dim=1, keepdim=True)+1e-8) # (bs, k)
        return topk, weight

    # deprecated, not stronger than the heatmap method
    def select_topk_object_by_segm(self, **kwargs):
        pose6d = kwargs['pose6d'].clone().float()
        pose6d[..., 6:] = pose6d[..., 6:] + kwargs['root_joint'][:, None]

        vert3d = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts')
        vert3d = self.obj_layer.flip_pt3d(vert3d, kwargs['is_right'])
        vert2d = project_point_by_cam_intrinsic(pt3d_cam=vert3d, cam_intrinsic=kwargs['cam_intrinsic'])

        bbox = kwargs['bbox'][:, None, None, :]
        bs, J, H, W = kwargs['segm'].shape
        vert2d = vert2d - bbox[..., :2]
        vert2d = vert2d / (bbox[..., 2:] - bbox[..., :2])
        vert2d[..., 0] = vert2d[..., 0] * (W - 1)
        vert2d[..., 1] = vert2d[..., 1] * (H - 1)

        vert2d = vert2d.round().long() # (bs, sample_num, N, 2)
        in_mask = (vert2d[..., 0] >= 0) & (vert2d[..., 0] <= W-1) & (vert2d[..., 1] >= 0) & (vert2d[..., 1] <= H-1)
        vert2d[~in_mask] = vert2d[~in_mask] *0 + torch.tensor([W, H-1], device=vert2d.device, dtype=vert2d.dtype)

        score_map = kwargs['segm'].repeat(1, kwargs['pose6d'].size(1), 1, 1) # (bs, sample_num, H, W)
        tmp_score = score_map.sum(dim=-1).sum(dim=-1) * -1 # (bs, sample_num)

        score_map = torch.cat([score_map, torch.zeros_like(score_map[:, :, :, :1])], dim=-1) # (bs, sample_num, H, W+1) add a column for out of bound
        
        idx_tensor1 = torch.arange(bs, device=score_map.device)[:, None, None].repeat(1, vert2d.size(1), vert2d.size(2))
        idx_tensor2 = torch.arange(vert2d.size(1), device=score_map.device)[None, :, None].repeat(bs, 1, vert2d.size(2))


        tmp_proj_mask = torch.zeros_like(score_map)

        score_map[idx_tensor1, idx_tensor2, vert2d[..., 1], vert2d[..., 0]] = 0
        score_map = score_map[:, :, :, :-1] # (bs, sample_num, H, W)
        score = score_map.sum(dim=-1).sum(dim=-1) * -1 # (bs, sample_num)

        val, topk = score.topk(kwargs['k'], dim=1)

        # # region [viz]
        # from lib.utils.viz_fn import depth_to_rgb
        # import numpy as np
        # import cv2

        # gt_mask = kwargs['segm'][0, 0]
        # gt_mask_rgb = depth_to_rgb(gt_mask)

        # tmp_proj_mask[idx_tensor1, idx_tensor2, vert2d[..., 1], vert2d[..., 0]] = 1
        # tmp_proj_mask = tmp_proj_mask[0, 0, :, :-1]
        # print(tmp_proj_mask.shape)
        # tmp_proj_mask = depth_to_rgb(tmp_proj_mask)

        # score_map = score_map[0, 0]
        # score_map = depth_to_rgb(score_map)

        # save_img = np.concatenate([gt_mask_rgb, score_map, tmp_proj_mask], axis=1)
        # cv2.imwrite("tmp.jpg", save_img)
        # print('ori: ', tmp_score[0, 0], 'score: ', score[0, 0])
        # print(vert2d[0, 0, :, 0].max(), vert2d[0, 0, :, 0].min())
        # print(vert2d[0, 0, :, 1].max(), vert2d[0, 0, :, 1].min())
        # # endregion
        return topk


    # deprecated, useless
    def __improve_obj_inplane_translation_using_heatmap(self, **kwargs):
        """ offset object inplane translation using the 2D center point from heatmap
            little improvement for the object translation, deprecated.
        """
        if kwargs['heatmap'].size(1) == 27:
            ct_hm = kwargs['heatmap'][:, 13]
            ct3d = kwargs['pt3d'][:, :, 13] # (bs, n, 3)
        elif kwargs['heatmap'].size(1) == 1:
            ct_hm = kwargs['heatmap'][:, 0]
            ct3d = kwargs['pt3d'][:, :, 0]
        else:
            raise NotImplementedError
        
        ct_hm = ct_hm.view(ct_hm.size(0), -1)
        maxvals, idx = torch.max(ct_hm, 1)
        ct2d = idx[:, None].repeat(1, 2)
        ct2d[:, 0] = ct2d[:, 0] % kwargs['heatmap'].size(3)
        ct2d[:, 1] = ct2d[:, 1] // kwargs['heatmap'].size(2)
        ct2d = ct2d.float() # (bs, 2)
        bbox = kwargs['bbox']
        ct2d = ct2d / (kwargs['heatmap'].size(3) - 1) * (bbox[:, 2:] - bbox[:, :2]) + kwargs['bbox'][:, :2] # (bs, 2)

        ct_uvd_hm = ct2d[:, None].repeat(1, ct3d.size(1), 1)
        ct_uvd_hm = torch.cat([ct_uvd_hm, ct3d[:, :, 2:]], dim=-1) # (bs, n, 3)
        ct3d_hm = inverse_project_point_by_cam_intrinsic(ct_uvd_hm, kwargs['cam_intrinsic']) # (bs, n, 3)

        weight = maxvals
        weight = weight[:, None, None]
        weight = weight
        weight = torch.clip(weight, 0, 1.0)

        ct3d_imporve = weight * ct3d_hm + (1 - weight) * ct3d
        # ct3d_imporve = ct3d_hm

        offset = ct3d_imporve - ct3d    
        return offset
    
    def search_along_z_axis(self, search_range=(-0.04, 0.04), **kwargs):
        """ search along the z-axis to find the best object pose
            pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            cam_intrinsic: (bs, 3, 3)
            heatmap: (bs, 21, H, W)
            bbox: (bs, 4)
            k: int
            is_right: (bs, )
            observe_index: (m,)
        """
        pose6d = kwargs['pose6d'].clone().float()
        root_joint = kwargs["root_joint"].unsqueeze(1)
        pose6d[..., 6:] =  pose6d[..., 6:] + root_joint
        obj_pt3d = self.obj_layer(pose6d, kwargs['obj_name'])
        obj_pt3d = self.obj_layer.flip_pt3d(obj_pt3d, kwargs['is_right'])

        obj_ct = obj_pt3d[..., 13, :] # (bs, 1, 3)
        ray = obj_ct / (obj_ct.norm(dim=-1, keepdim=True)+1e-8)

        search_points = torch.arange(search_range[0], search_range[1]+1e-8, step=(search_range[1]-search_range[0])/100, device=pose6d.device)
        search_points = search_points[:, None] * ray

        pose6d_search = kwargs['pose6d'].clone().float()
        pose6d_search = pose6d_search.repeat(1, search_points.size(1), 1)
        pose6d_search[..., 6:] = pose6d_search[..., 6:] + search_points
        kwargs['pose6d'] = pose6d_search
        kwargs['k'] = 1
        topk = self.select_topk_object_by_heatmap(**kwargs)
        return topk, pose6d_search
    
    def select_topk_object_by_physics(self, **kwargs):
        """ Rank the object candidates by the torque

            pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            is_right: (bs, )
            force_point: (bs, 32, 3)
            force_global: (bs, 32, 3)
        """
        pose6d = kwargs['pose6d'].clone().float()
        root_joint = kwargs["root_joint"].unsqueeze(1)
        pose6d[..., 6:] =  pose6d[..., 6:] + root_joint
        obj_CoM = self.obj_layer(pose6d, kwargs['obj_name'], data_name='CoM') # (bs, n_candidate, 1, 3)
        obj_CoM = self.obj_layer.flip_pt3d(obj_CoM, kwargs['is_right'])
        
        force_point = kwargs['force_point'] + root_joint # (bs, 32, 3)
        force_global = kwargs['force_global'] # (bs, 32, 3)

        arm = force_point[:, None] - obj_CoM # (bs, n_candidate, 32, 3)
        torque = torch.cross(arm, force_global[:, None], dim=-1).sum(-2)
        torque = torch.norm(torque, dim=-1) # (bs, n_candidate)
        val, topk = (-torque).topk(kwargs['k'], dim=1)
        return topk

        # region [viz]
        # obj_verts = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts')
        # obj_verts = self.obj_layer.flip_pt3d(obj_verts, kwargs['is_right'])
        # force = torch.stack([force_point, force_point+force_global*0.3], dim=-2)
        # save_dt = {
        #     'force_#FF0000': force[0].detach().cpu().numpy(),
        #     'obj_vert_#00FF00': obj_verts[0, 0].detach().cpu().numpy()
        # }
        # import pickle
        # with open(f"tmp_{obj_verts.device}.pkl", "wb") as f:
        #     pickle.dump(save_dt, f)
        # exit()
        # endregion


    def select_topk_object_by_physics2(self, **kwargs):
        """ Rank the object candidates by the weighed distance of force point
            weights = |force_global| / sum(|force_global|)

            pose6d: (bs, n, 9)
            root_joint: (bs, 3)
            obj_name: (bs, )
            is_right: (bs, )
            force_point: (bs, 32, 3)
            force_global: (bs, 32, 3)
        """
        pose6d = kwargs['pose6d'].clone().float()
        root_joint = kwargs["root_joint"].unsqueeze(1)
        pose6d[..., 6:] =  pose6d[..., 6:] + root_joint
        obj_verts = self.obj_layer(pose6d, kwargs['obj_name'], data_name='verts') # (bs, n_candidate, 2048, 3)
        obj_verts = self.obj_layer.flip_pt3d(obj_verts, kwargs['is_right']) 
        
        force_point = kwargs['force_point'] # (bs, 32, 3)
        force_global = kwargs['force_global'] # (bs, 32, 3)
        force_global_norm = force_global.norm(dim=-1) # (bs, 32)
        force_weight = force_global_norm / force_global_norm.sum(dim=-1, keepdim=True) # (bs, 32)

        # # region [tmp]
        # idx_tensor = torch.arange(force_point.size(0), device=force_point.device)
        # idx_tensor = idx_tensor[~kwargs['is_right']]
        # print(kwargs['is_right'].sum(), idx_tensor.shape)
        # if idx_tensor.size(0) != 0:
        #     force = torch.stack([force_point, force_point+force_global*0.3], dim=-2)
        #     save_dt = {
        #         'force_#FF0000': force[idx_tensor[0]].detach().cpu().numpy(),
        #         'obj_vert_#00FF00': obj_verts[idx_tensor[0], 0].detach().cpu().numpy(),
        #         'hand_vert_#000000': kwargs['hand_vert'][idx_tensor[0]].detach().cpu().numpy(),
        #     }
        #     import pickle
        #     with open(f"tmp_{obj_verts.device}.pkl", "wb") as f:
        #         pickle.dump(save_dt, f)   
        # # endregion

        force_point = force_point[:, None] # (bs, 1, 32, 3)
        cdist = cdist_memory_save(force_point, obj_verts) # (bs, n_candidate, 32)

        score = -(cdist * force_weight[:, None]).sum(-1)
        val, topk = score.topk(kwargs['k'], dim=1)
        weight = torch.ones_like(val)
        weight = weight / weight.sum(dim=1, keepdim=True)
        return topk, weight



def cdist_memory_save(x, y):
    """ x: (bs, n, v, d)
        y: (bs, n, v, d)
    """
    cdist_ls = []
    for i in range(x.size(0)):
        cdist_i = torch.cdist(x[i], y[i], p=2)
        cdist_i = cdist_i.min(dim=-1)[0]
        cdist_ls.append(cdist_i)
    cdist = torch.stack(cdist_ls, dim=0)
    return cdist



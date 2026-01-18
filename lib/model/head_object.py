import torch
import torch.nn as nn
from pytorch3d.transforms.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix

from lib.dataset.base import YCB_MESHES
from lib.utils.transform_fn import matmul_for_rt


class HeadObject(nn.Module):
    def __init__(self, ):
        super(HeadObject, self).__init__()

        self.obj_mesh = {}
        for k in YCB_MESHES.keys():
            kpt3d = YCB_MESHES[k]['kpt3d']
            kpt3d = torch.from_numpy(kpt3d).float()
            self.register_buffer(f"point_{k}", kpt3d)
            
            shift = YCB_MESHES[k]['shift']
            shift = torch.tensor(shift).float()
            # self.register_buffer(f"shift_{k}", shift)  # TODO: use axsym
            
            vert3d = YCB_MESHES[k]['verts_sampled']
            vert3d = torch.from_numpy(vert3d).float()
            self.register_buffer(f"vert_{k}", vert3d)

            CoM = YCB_MESHES[k]['CoM']
            CoM = torch.tensor(CoM).float()[None]
            self.register_buffer(f"CoM_{k}", CoM)

            vert3d = YCB_MESHES[k]['verts']
            vert3d = torch.from_numpy(vert3d).float()
            self.register_buffer(f"vert_full_{k}", vert3d)


    def forward(self, pose, name, data_name='keypoint'):
        """ pose: (bs, ..., 9)
            name: (bs, )
            data_name: in ['keypoint', 'verts', 'CoM', 'verts_full]
        """
        pts_ls = []
        for n in name:
            # vert = self.obj_mesh[n]
            if data_name == 'keypoint':
                vert = getattr(self, f"point_{n}")
            elif data_name == 'verts':
                vert = getattr(self, f"vert_{n}")
            elif data_name == 'verts_full':
                vert = getattr(self, f"vert_full_{n}")
            elif data_name == 'CoM':
                vert = getattr(self, f"CoM_{n}")
            else:
                raise ValueError(f"Unknown data_name: {data_name}")
            pts_ls.append(vert)
        pts = torch.stack(pts_ls, dim=0)

        rotmat = rotation_6d_to_matrix(pose[..., :6])
        new_verts = torch.einsum('bvi,b...ji->b...vj', pts, rotmat)
        trans = pose[..., 6:].unsqueeze(-2)
        new_verts = new_verts + trans
        return new_verts
    
    def flip_pt3d(self, pt3d, is_right):
        idx_tensor = torch.arange(pt3d.shape[0], device=pt3d.device)
        flipped_idx = idx_tensor[~is_right]
        pt3d[flipped_idx, ..., 0] = pt3d[flipped_idx, ..., 0] * -1
        return pt3d

    def to_axsym_pose(self, pose, name):
        """ pose: (bs, ..., 6)
            name: (bs, )
        """
        shift_ls = []
        for n in name:
            shift = getattr(self, f"shift_{n}")
            shift_ls.append(shift)
        shift = torch.stack(shift_ls, dim=0) # (bs, 3, 4)
        shift_r = shift[..., :3, :3]
        shift_t = shift[..., :3, 3]

        axsym_to_cam_r = shift_r.transpose(1, 2)
        axsym_to_cam_t = -torch.einsum('b...ij,bj->b...i', axsym_to_cam_r, shift_t)
        axsym_to_cam_rt = torch.cat([axsym_to_cam_r, axsym_to_cam_t.unsqueeze(-1)], dim=-1)

        obj_r = rotation_6d_to_matrix(pose[..., :6])
        obj_t = pose[..., 6:]
        obj_rt = torch.cat([obj_r, obj_t.unsqueeze(-1)], dim=-1)

        new_rt = matmul_for_rt(obj_rt, axsym_to_cam_rt)
        new_r = new_rt[..., :3, :3]
        new_t = new_rt[..., :3, 3]

        new_r6d = matrix_to_rotation_6d(new_r)
        new_pose = torch.cat([new_r6d, new_t], dim=-1)
        return new_pose

    def to_cam_pose(self, pose, name):
        """ pose: (bs, ..., 6)
            name: (bs, )
        """
        shift_ls = []
        for n in name:
            shift = getattr(self, f"shift_{n}")
            shift_ls.append(shift)
        shift = torch.stack(shift_ls, dim=0)

        cam_to_axsym_r = shift[..., :3, :3]
        cam_to_axsym_t = shift[..., :3, 3]
        cam_to_axsym_rt = torch.cat([cam_to_axsym_r, cam_to_axsym_t.unsqueeze(-1)], dim=-1)

        obj_r = rotation_6d_to_matrix(pose[..., :6])
        obj_t = pose[..., 6:]
        obj_rt = torch.cat([obj_r, obj_t.unsqueeze(-1)], dim=-1)

        new_rt = matmul_for_rt(obj_rt, cam_to_axsym_rt)
        new_r = new_rt[..., :3, :3]
        new_t = new_rt[..., :3, 3]

        new_r6d = matrix_to_rotation_6d(new_r)
        new_pose = torch.cat([new_r6d, new_t], dim=-1)
        return new_pose

    

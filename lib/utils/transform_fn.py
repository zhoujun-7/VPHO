import torch
import numpy as np
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


def project_pt3d_to_pt2d(pt3d, cam_intrinsic):
    """ pt3d: (..., 3)
        cam_intrinsic: (..., 3, 3)
    """
    if isinstance(pt3d, np.ndarray):
        pt2d = np.matmul(pt3d, cam_intrinsic.transpose(-1, -2))
        pt2d = pt2d / pt2d[..., -1:]
        pt2d = pt2d[..., :-1]
    elif isinstance(pt3d, torch.Tensor):
        pt2d = torch.matmul(pt3d, cam_intrinsic.transpose(-1, -2))
        pt2d = pt2d / pt2d[..., -1:]
        pt2d = pt2d[..., :-1]
    else:
        raise NotImplementedError
    return pt2d


def inverse_project_uvd_to_xyz(uvd, cam_intrinsic):
    """ uvd: (..., 3)
        cam_intrinsic: (..., 3, 3)
    """
    if isinstance(uvd, np.ndarray):
        xyz = np.ones_like(uvd)
        xyz[..., :-1] = uvd[..., :-1]
        xyz = np.matmul(xyz, np.linalg.inv(cam_intrinsic).transpose(-1, -2))
        xyz *= uvd[..., [-1]]
    elif isinstance(uvd, torch.Tensor):
        xyz = torch.ones_like(uvd)
        xyz[..., :-1] = uvd[..., :-1]
        xyz = torch.matmul(xyz, torch.inverse(cam_intrinsic).transpose(-1, -2))
        xyz *= uvd[..., [-1]]
    else:
        raise NotImplementedError
    return xyz


#* from 2023_CVPR_HFL
def rigid_transform_3D_AtoB(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s) 

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


#* from 2023_CVPR_HFL
def rigid_align_AtoB(A,B):
    c, R, t = rigid_transform_3D_AtoB(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2


def depth_to_uvd(depth, background_val=0):
    if isinstance(depth, np.ndarray):
        h, w = depth.shape
        X = np.arange(w)
        Y = np.arange(h)
        XX, YY = np.meshgrid(X, Y)
        fore_mask = depth != background_val
        u = XX[fore_mask][:, None]
        v = YY[fore_mask][:, None]
        d = depth[fore_mask][:, None]
        uvd = np.concatenate([u, v, d], axis=1)  # n, 3
    else:
        raise NotImplementedError
    return uvd


def obj_9D_to_mat(obj_9D):
    assert isinstance(obj_9D, torch.Tensor)
    obj_rot6d = obj_9D[..., :6]
    obj_rotmat = rotation_6d_to_matrix(obj_rot6d)
    obj_rt = torch.cat([obj_rotmat, obj_9D[..., 6:9, None]], dim=-1)
    return obj_rt

def obj_mat_to_9D(obj_rt):
    assert isinstance(obj_rt, torch.Tensor)
    obj_rotmat = obj_rt[..., :3, :3]
    obj_rot6d = matrix_to_rotation_6d(obj_rotmat)
    obj_9D = torch.cat([obj_rot6d, obj_rt[..., :3, 3]], dim=-1)
    return obj_9D


# modified from 2023_NIPS_GenPose, checked.
def average_quaternion(Q, W=None):
    """calculate the average quaternion of the multiple quaternions along the -2 dimension
    Args:
        Q: (B, ..., N, 4)
        weights: (B, ..., N). Defaults to None.
    Returns:
        oriented_q_avg: average quaternion, (B, ..., 4)
    """
    shape = Q.shape
    assert shape[-1] == 4

    if W is None:
        W = torch.ones_like(Q[..., 0])
    else:
        assert shape[:-1] == W.shape

    weight_sum = W.sum(dim=-1, keepdim=True)
    oriented_Q = ((Q[..., 0:1] > 0).float() - 0.5) * 2 * Q
    A = torch.einsum("...ni,...nj->...nij", oriented_Q, oriented_Q)
    A = torch.sum(torch.einsum("...nij,...n->...nij", (A, W)), -3)
    A /= weight_sum.reshape(*shape[:-2], 1, 1)

    q_avg = torch.linalg.eigh(A)[1][..., -1]
    oriented_q_avg = ((q_avg[..., 0:1] > 0).float() - 0.5) * 2 * q_avg
    return oriented_q_avg


def matmul_for_rt(T1, T2):
    """ T1: (..., 3, 4)
        T2: (..., 3, 4)
    """
    if isinstance(T1, torch.Tensor):
        r1 = T1[..., :3, :3]
        t1 = T1[..., :3, 3]
        r2 = T2[..., :3, :3]
        t2 = T2[..., :3, 3]

        new_r = torch.einsum('b...ij,b...jk->b...ik', r1, r2)
        new_t = torch.einsum('b...ij,b...j->b...i', r1, t2) + t1
        new_rt = torch.cat([new_r, new_t[..., None]], dim=-1)
    elif isinstance(T1, np.ndarray):
        r1 = T1[..., :3, :3]
        t1 = T1[..., :3, 3:]
        r2 = T2[..., :3, :3]
        t2 = T2[..., :3, 3:]

        new_r = np.einsum('...ij,...jk->...ik', r1, r2)
        new_t = np.einsum('...ij,...j->...i', r1, t2) + t1
        new_rt = np.concatenate([new_r, new_t[..., None]], axis=-1)
    else:
        raise NotImplementedError
    return new_rt



OPENGL_TO_OPENCV = np.array([[1, 0, 0], 
                             [0, -1, 0], 
                             [0, 0, -1]])
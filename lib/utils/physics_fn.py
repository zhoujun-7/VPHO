import time
import numpy as np
import torch
import os
import pickle
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from collections import OrderedDict

from .hand_fn import SKELETON, SKELETON_LEVEL, VERT2JOINT


def detect_thin_shell(verts, normals, shell_thickness_thresh, normal_angle_thresh):
    """
    Detect thin shell in the input mesh.
    Args:
        verts   (N, 3): vertices of the input mesh
        normals (N, 3): faces of the input mesh
        shell_thickness_thresh (2,): threshold for shell thickness in m
        normal_angle_thresh (float): threshold for normal angle in degree
    """
    nbrs = NearestNeighbors(n_neighbors=21, algorithm='ball_tree').fit(verts)
    distances, indices = nbrs.kneighbors(verts)

    thin_shell_ls = []
    for i in range(verts.shape[0]):
        normal_i = normals[i]
        dist = verts[indices[i, 1:]] - verts[i]
        dist_along_normal = np.abs(np.dot(dist, normal_i))
        in_shell_thresh_mask = np.logical_and(dist_along_normal > shell_thickness_thresh[0], dist_along_normal < shell_thickness_thresh[1])

        for j in range(1, 11):
            if not in_shell_thresh_mask[j]:
                continue
            normal_j = normals[indices[i, j]]
            angle = np.arccos(np.dot(normal_i, normal_j) / (np.linalg.norm(normal_i) * np.linalg.norm(normal_j)))
            angle = np.rad2deg(angle)
            if angle > normal_angle_thresh:
                pair = (i, indices[i, j])
                thin_shell_ls.append(pair)
                break

    thin_shell_ls = np.array(thin_shell_ls)
    return thin_shell_ls


def detect_hand_and_object_contact(hand_verts, hand_normals, obj_verts, obj_normals, normal_distance_thresh=[-0.015, 0.01], vertical_distance_thresh=0.01, decay_points=[-0.005, 0.005]):
    """
    Detect contact between hand and object.
    Args:
        hand_verts (..., N, 3): vertices of the hand mesh
        hand_normals (..., N, 3): normals of the hand mesh
        obj_verts (..., N, 3): vertices of the object mesh
        obj_normals (..., N, 3): normals of the object mesh
        normal_distance_thresh (2,): threshold for distance along normal in m
        vertical_distance_thresh (float): threshold for distance on the plane vertical to normal in m
    Returns:
        hand_contact_map (..., N): contact map for hand mesh
        obj_contact_map (..., N): contact map for object mesh
        obj_contact_to_hand_vert (..., N): mapping from object contact vertices to hand vertices
    """
    assert normal_distance_thresh[0] < normal_distance_thresh[1]
    assert decay_points[0] < decay_points[1]
    assert vertical_distance_thresh > 0
    assert normal_distance_thresh[0] < decay_points[0] < decay_points[1] < normal_distance_thresh[1]

    if isinstance(hand_verts, np.ndarray):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')

        nbrs.fit(obj_verts)
        h2o_dist, h2o_ind = nbrs.kneighbors(hand_verts)
        h2o_dist, h2o_ind = h2o_dist.squeeze(), h2o_ind.squeeze()
        h2o_vec = hand_verts - obj_verts[h2o_ind.squeeze()]
        h2o_normal_dist = np.sum(h2o_vec * hand_normals, axis=-1)
        h2o_vertical_dist = np.linalg.norm(h2o_vec - h2o_normal_dist[..., None] * hand_normals, axis=-1)
        hand_contact_mask1 = np.logical_and(h2o_normal_dist > normal_distance_thresh[0], h2o_normal_dist < normal_distance_thresh[1])
        hand_contact_mask2 = h2o_vertical_dist < vertical_distance_thresh
        hand_contact_mask = np.logical_and(hand_contact_mask1, hand_contact_mask2)
        hand_contact_map = h2o_normal_dist.copy()

        nbrs.fit(hand_verts)
        o2h_dist, o2h_ind = nbrs.kneighbors(obj_verts)
        o2h_dist, o2h_ind = o2h_dist.squeeze(), o2h_ind.squeeze()
        o2h_vec = obj_verts - hand_verts[o2h_ind.squeeze()]
        o2h_normal_dist = np.sum(o2h_vec * obj_normals, axis=-1)
        o2h_vertical_dist = np.linalg.norm(o2h_vec - o2h_normal_dist[..., None] * obj_normals, axis=-1)
        obj_contact_mask1 = np.logical_and(o2h_normal_dist > normal_distance_thresh[0], o2h_normal_dist < normal_distance_thresh[1])
        obj_contact_mask2 = o2h_vertical_dist < vertical_distance_thresh
        obj_contact_mask = np.logical_and(obj_contact_mask1, obj_contact_mask2)
        obj_contact_map = o2h_normal_dist.copy()

        obj_contact_to_hand_vert = np.ones_like(obj_contact_map, dtype=np.int32) * -1
        obj_contact_to_hand_vert[obj_contact_mask] = o2h_ind[obj_contact_mask]
        
        # region [contact distance to weight]
        mid_point1 = (decay_points[0]+normal_distance_thresh[0])/2
        mid_point2 = (decay_points[1]+normal_distance_thresh[1])/2
        def contact_weight_fn(x):
            map1 = 1 + np.exp(-1600 * (x - mid_point1))
            w1 = np.isfinite(map1)
            map2 = 1 + np.exp(1600 * (x - mid_point2))
            w2 = np.isfinite(map2)
            map3 = 1 / (map1 * map2 + 1e-10)
            map3[~w1] = 0
            map3[~w2] = 0
            return map3
        scale = contact_weight_fn(np.array([0])) # scale max contact value to 1
        hand_contact_map = contact_weight_fn(hand_contact_map) / scale
        obj_contact_map = contact_weight_fn(obj_contact_map) / scale
        hand_contact_map[~hand_contact_mask] = 0
        obj_contact_map[~obj_contact_mask] = 0
        # endregion

    else:
        raise NotImplementedError
    
    return hand_contact_map, obj_contact_map, obj_contact_to_hand_vert


#* modified from 2021_CVPR_CPF
class ForceAnchor:
    def __init__(self, assert_path="asset/2021_CVPR_CPF"):
        self.face_vert_idx, self.anchor_weight, self.merged_vertex_assignment, self.anchor_mapping = self.anchor_load_driver(assert_path)
        self.anchor_weight = np.concatenate([np.ones([self.anchor_weight.shape[0], 1]), self.anchor_weight], axis=1)
        self.anchor_weight_pth = torch.from_numpy(self.anchor_weight).float()

        self.label_level = OrderedDict()
        self.label_level["WIM"] = [5] # Wrist to Index Metacarpal
        self.label_level["WMM"] = [12]
        self.label_level["WRM"] = [19, 18]
        self.label_level["WPM"] = [26, 25]
        
        self.label_level["MTP"] = [6, 0] # Thumb Metacarpal to Proximal
        self.label_level["MIP"] = [7]
        self.label_level["MMP"] = [13]
        self.label_level["MRP"] = [20]
        self.label_level["MPP"] = [27]

        self.label_level["PTD"] = [1] # Proximal to Distal
        self.label_level["PID"] = [8]
        self.label_level["PMD"] = [14]
        self.label_level["PRD"] = [21]
        self.label_level["PPD"] = [28]

        self.label_level["DTT"] = [2, 3, 4] # Thumb Distal to Tip, inverse clock-wise from tip 
        self.label_level["DIT"] = [9, 11, 10] 
        self.label_level["DMT"] = [15, 17, 16]
        self.label_level["DRT"] = [22, 24, 23]
        self.label_level["DPT"] = [29, 31, 30]
        
        self.label = []
        for k, v in self.label_level.items():
            self.label.extend(v)
        self.label = np.array(self.label)
        
        self.coresponding_skeleton = [
            SKELETON_LEVEL[0][1], SKELETON_LEVEL[0][2], SKELETON_LEVEL[0][3], SKELETON_LEVEL[0][3], SKELETON_LEVEL[0][4], SKELETON_LEVEL[0][4],
            SKELETON_LEVEL[0][0], SKELETON_LEVEL[0][0], SKELETON_LEVEL[1][1], SKELETON_LEVEL[1][2], SKELETON_LEVEL[1][3], SKELETON_LEVEL[1][4],
            SKELETON_LEVEL[2][0], SKELETON_LEVEL[2][1], SKELETON_LEVEL[2][2], SKELETON_LEVEL[2][3], SKELETON_LEVEL[2][4], 

            SKELETON_LEVEL[3][0], SKELETON_LEVEL[3][0], SKELETON_LEVEL[3][0],
            SKELETON_LEVEL[3][1], SKELETON_LEVEL[3][1], SKELETON_LEVEL[3][1],
            SKELETON_LEVEL[3][2], SKELETON_LEVEL[3][2], SKELETON_LEVEL[3][2],
            SKELETON_LEVEL[3][3], SKELETON_LEVEL[3][3], SKELETON_LEVEL[3][3],
            SKELETON_LEVEL[3][4], SKELETON_LEVEL[3][4], SKELETON_LEVEL[3][4],
        ]
        self.coresponding_skeleton = np.array(self.coresponding_skeleton)
        order_back_idx = np.argsort(self.label)
        self.coresponding_skeleton = self.coresponding_skeleton[order_back_idx]

        self.finger_label = OrderedDict()
        self.finger_label['palm'] = [] + self.label_level["WIM"] + self.label_level["WMM"] + self.label_level["WRM"] + self.label_level["WPM"]
        self.finger_label['thumb'] = [] + self.label_level["MTP"] + self.label_level["PTD"] + self.label_level["DTT"]
        self.finger_label['index'] = [] + self.label_level["MIP"] + self.label_level["PID"] + self.label_level["DIT"]
        self.finger_label['middle'] = [] + self.label_level["MMP"] + self.label_level["PMD"] + self.label_level["DMT"]
        self.finger_label['ring'] = [] + self.label_level["MRP"] + self.label_level["PRD"] + self.label_level["DRT"]
        self.finger_label['pinky'] = [] + self.label_level["MPP"] + self.label_level["PPD"] + self.label_level["DPT"]


    def anchor_load_driver(self, inpath):
        anchor_root = os.path.join(inpath, "anchor")
        face_vert_idx, anchor_weight, merged_vertex_assignment, anchor_mapping = self.anchor_load(anchor_root)
        return face_vert_idx, anchor_weight, merged_vertex_assignment, anchor_mapping

    def anchor_load(self, anchor_root):
        # face vert idx
        face_vert_idx_path = os.path.join(anchor_root, "face_vertex_idx.txt")
        face_vert_idx = np.loadtxt(face_vert_idx_path, dtype=np.int32)
        # anchor weight
        anchor_weight_path = os.path.join(anchor_root, "anchor_weight.txt")
        anchor_weight = np.loadtxt(anchor_weight_path)
        # vertex assignment
        vertex_assigned_path = os.path.join(anchor_root, "merged_vertex_assignment.txt")
        merged_vertex_assignment = np.loadtxt(vertex_assigned_path, dtype=np.int32)
        # load the anchor mapping
        anchor_mapping_path = os.path.join(anchor_root, "anchor_mapping_path.pkl")
        with open(anchor_mapping_path, "rb") as fstream:
            anchor_mapping = pickle.load(fstream)
        return face_vert_idx, anchor_weight, merged_vertex_assignment, anchor_mapping

    def get_force_contact(self, hand_contact):
        force_contact = hand_contact[..., self.face_vert_idx.reshape(-1)].reshape(hand_contact.shape[:-1] + (-1, 3))
        if isinstance(hand_contact, torch.Tensor):
            force_contact = force_contact * (self.anchor_weight_pth / self.anchor_weight_pth.sum(dim=1, keepdim=True))
        elif isinstance(hand_contact, np.ndarray):
            force_contact = force_contact * (self.anchor_weight / self.anchor_weight.sum(axis=1, keepdims=True))
        force_contact = force_contact.sum(-1)
        return force_contact
    
    def check_is_grasped(self, force_contact, thresh=0):
        # force_contact: (..., 32)
        mask_palm = force_contact[..., self.finger_label['palm']].sum(-1) > thresh
        mask_thumb = force_contact[..., self.finger_label['thumb']].sum(-1) > thresh
        mask_index = force_contact[..., self.finger_label['index']].sum(-1) > thresh
        mask_middle = force_contact[..., self.finger_label['middle']].sum(-1) > thresh
        mask_ring = force_contact[..., self.finger_label['ring']].sum(-1) > thresh
        mask_pinky = force_contact[..., self.finger_label['pinky']].sum(-1) > thresh

        is_grasped = np.stack([mask_palm, mask_thumb, mask_index, mask_middle, mask_ring, mask_pinky], axis=-1)
        is_grasped = is_grasped.sum() >= 2
        return is_grasped

    
    def __call__(self, vertices):
        # verts: (..., N, 3)
        face_vert_idx = self.face_vert_idx.reshape(-1)
        indexed_vertices = vertices[..., face_vert_idx, :].reshape(vertices.shape[:-2] + (-1, 3, 3))
        base_vec_1 = indexed_vertices[..., :, 1, :] - indexed_vertices[..., :, 0, :]
        base_vec_2 = indexed_vertices[..., :, 2, :] - indexed_vertices[..., :, 0, :]
        joints = VERT2JOINT(vertices)

        anchor_direction_y = joints[..., self.coresponding_skeleton[:, 1], :] - joints[..., self.coresponding_skeleton[:, 0], :]

        if isinstance(vertices, torch.Tensor):
            self.anchor_weight_pth = self.anchor_weight_pth.to(vertices.device)
            anchor_weight = self.anchor_weight_pth
            anchor_direction_z = torch.cross(base_vec_1, base_vec_2, dim=-1)
            anchor_direction_z = anchor_direction_z / (torch.norm(anchor_direction_z, dim=-1, keepdim=True) + 1e-8)
            anchor_direction_y = anchor_direction_y / (torch.norm(anchor_direction_y, dim=-1, keepdim=True) + 1e-8)
            anchor_direction_x = torch.cross(anchor_direction_y, anchor_direction_z, dim=-1)
            anchor_direction_y = torch.cross(anchor_direction_z, anchor_direction_x, dim=-1)
            anchor_direction_y = anchor_direction_y / (torch.norm(anchor_direction_y, dim=-1, keepdim=True) + 1e-8)
            anchor_frame = torch.stack([anchor_direction_x, anchor_direction_y, anchor_direction_z], dim=-1)
        elif isinstance(vertices, np.ndarray):
            anchor_weight = self.anchor_weight
            anchor_direction_z = np.cross(base_vec_1, base_vec_2, axis=-1)
            anchor_direction_z = anchor_direction_z / (np.linalg.norm(anchor_direction_z, axis=-1, keepdims=True) + 1e-8)
            anchor_direction_y = anchor_direction_y / (np.linalg.norm(anchor_direction_y, axis=-1, keepdims=True) + 1e-8)
            anchor_direction_x = np.cross(anchor_direction_y, anchor_direction_z, axis=-1)
            anchor_direction_y = np.cross(anchor_direction_z, anchor_direction_x, axis=-1)
            anchor_direction_y = anchor_direction_y / (np.linalg.norm(anchor_direction_y, axis=-1, keepdims=True) + 1e-8)
            anchor_frame = np.stack([anchor_direction_x, anchor_direction_y, anchor_direction_z], axis=-1)

        rebuilt_anchors = anchor_weight[:, 1:2] * base_vec_1 + anchor_weight[:, 2:3] * base_vec_2
        origins = indexed_vertices[..., :, 0, :]
        rebuilt_anchors = rebuilt_anchors + origins
        return rebuilt_anchors, anchor_frame
VERT2ANCHOR = ForceAnchor()

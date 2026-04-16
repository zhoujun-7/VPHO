import os
import numpy as np
import pickle
import time
import torch
import tqdm
import trimesh
from collections import defaultdict
import pandas
import warnings

from lib.configs.args import cfg
from lib.dataset.base import YCB_MESHES, mano_layer_r
from lib.utils.transform_fn import rigid_align_AtoB
from lib.model.physics import from_local_to_global
from lib.thirdparty.mujoco_sim.mojuco_sim import MuJoCoMeshSimulatorFast
from lib.thirdparty.libmesh.inside_mesh import check_mesh_contains

from kaolin.ops.mesh import check_sign, index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion


# from 2021-CVPR-CPF
def solid_intersection_volume(hand_verts, hand_faces, obj_vox_points, obj_tsl, obj_rot, obj_vox_el_vol):
    # first transf points to desired location
    # convert obj_rot to rotation matrix
    if obj_rot.shape == (3, 3):
        obj_rotmat = obj_rot
    obj_vox_points_transf = (obj_rotmat @ obj_vox_points.T).T
    obj_vox_points_transf = obj_vox_points_transf + obj_tsl
    # create hand trimesh
    hand_trimesh = trimesh.Trimesh(vertices=np.asarray(hand_verts), faces=np.asarray(hand_faces))
    # _ = hand_trimesh.vertex_normals
    # _ = hand_trimesh.face_normals
    # hand_trimesh.fix_normals()
    # inside = hand_trimesh.contains(obj_vox_points_transf)
    inside = check_mesh_contains(hand_trimesh, obj_vox_points_transf)
    volume = inside.sum() * obj_vox_el_vol
    return volume, obj_vox_points_transf, inside


# from 2023-NIPS-DeepSimHO
def calculate_sdf(hand_verts, obj_verts, hand_faces, obj_faces):
    # copy implementation from stabilityloss.calculate_sdf to avoid module conflict
    batch_size = hand_verts.shape[0]

    hand_sample_obj_mesh_pd_ = []

    for i in range(batch_size):
        obj_face = obj_faces[i]
        for k in range(obj_face.shape[0]-1, 0, -1):
            if not obj_face[k].eq(0).all():
                break
        obj_face = obj_face[:k+1, :]
        face_vertices = index_vertices_by_faces(obj_verts[i].unsqueeze(0).contiguous(), obj_face)
        distance, _, _ = point_to_mesh_distance(hand_verts[i].unsqueeze(0).contiguous(), face_vertices)
        sign = check_sign(obj_verts[i].unsqueeze(0).contiguous(), obj_face, hand_verts[i].unsqueeze(0).contiguous()).int()
        sign[sign == 0] = -1
        hand_sample_obj_mesh_pd_.append(distance * sign.int())

    ho_sdf = torch.stack(hand_sample_obj_mesh_pd_)

    object_sample_hand_mesh_pd_ = []

    for i in range(batch_size):
        hand_face = hand_faces[i]
        face_vertices = index_vertices_by_faces(hand_verts[i].unsqueeze(0).contiguous(), hand_face)
        distance, _, _ = point_to_mesh_distance(obj_verts[i].unsqueeze(0).contiguous(), face_vertices)
        sign = check_sign(hand_verts[i].unsqueeze(0).contiguous(), hand_face, obj_verts[i].unsqueeze(0).contiguous()).int()
        sign[sign == 0] = -1
        object_sample_hand_mesh_pd_.append(distance * sign.int())

    oh_sdf = torch.stack(object_sample_hand_mesh_pd_)

    sdf_dist = torch.cat([ho_sdf.squeeze(1), oh_sdf.squeeze(1)], dim=1) # 778 + 1000 shape

    return sdf_dist


class TesterPhysics:
    def __init__(self):
        self.cfg = cfg
        self.obj_mesh = YCB_MESHES

        self.g = 9.8
        self.SD_time = 0.2 # 200ms  #* align with 2023-NIPS-DeepSimHO
        self.SD_thresh = 0.01 # 1cm

        self.load_hand()
        self.load_obj()


    def load_hand(self,):
        self.hand_faces = mano_layer_r.th_faces.clone()
        hand_closed_trimesh = trimesh.load('asset/2021_CVPR_CPF/closed_hand/hand_mesh_close.obj', process=False)
        self.hand_close_faces_np = np.array(hand_closed_trimesh.faces)
    
    # modified from 2021-CVPR-CPF
    def load_obj(self,):
        obj_root = "asset/2021_CVPR_CPF/YCB_models_supp"
        object_names = [obj_name for obj_name in os.listdir(obj_root) if ".tgz" not in obj_name]
        objects = {}
        for obj_name in object_names:
            obj_path = os.path.join(obj_root, obj_name, "solid.binvox")
            vox = trimesh.load(obj_path)
            objects[obj_name] = {
                "points": np.array(vox.points),
                "matrix": np.array(vox.matrix),
                "element_volume": vox.element_volume,
            }
        self.obj_vox = objects
        return objects

    def eval_use_force(self, data):
        """ data: {
            'pd_hand_vert': (N, V, 3),
            'gt_hand_vert': (N, V, 3),
            'pd_obj_rt': (N, 3, 4),
            'gt_obj_rt': (N, 3, 4),
            'pd_force_local': (N, 32, 3),
            'gravity': (N, 1, 3),
            'obj_name': (N, ),
        """
        data['force_point'], data['force_global'] = from_local_to_global(data['pd_force_local'], data['pd_hand_vert'])

        length = data['gt_hand_vert'].shape[0]
        PD, CP, SD, SR, SIV, H_IF, O_IF = [], [], [], [], [], [], []
        for i in range(length):
            pd_hand_vert = data['pd_hand_vert'][i]
            gt_hand_vert = data['gt_hand_vert'][i]
            pd_obj_rt = data['pd_obj_rt'][i]
            gt_obj_rt = data['gt_obj_rt'][i]
            pd_force = data['force_global'][i]
            gravity = data['gravity'][i]
            obj_name = data['obj_name'][i]
            ori_obj_vert = self.obj_mesh[obj_name]['verts']
            obj_face = self.obj_mesh[obj_name]['faces']
            pd_obj_vert = ori_obj_vert @ pd_obj_rt[:, :3].T + pd_obj_rt[:, 3]
            gt_obj_vert = ori_obj_vert @ gt_obj_rt[:, :3].T + gt_obj_rt[:, 3]

            pd, cp = self.criterion_PD_CP(pd_hand_vert, pd_obj_vert, self.hand_faces, obj_face)
            sd, sr = self.criterion_SD_SR(pd_force, gravity, pd_obj_rt[:, 3], gt_obj_rt[:, 3])
            siv = self.criterion_SIV(pd_hand_vert, pd_obj_rt, obj_name)
            h_iF, o_iF = self.criterion_IF(pd_hand_vert, pd_obj_vert, gt_hand_vert, gt_obj_vert)

            PD.append(pd)
            CP.append(cp)
            SD.append(sd)
            SR.append(sr)
            SIV.append(siv)
            H_IF.append(h_iF)
            O_IF.append(o_iF)
        PD = np.stack(PD, axis=0)
        CP = np.stack(CP, axis=0)
        SD = np.stack(SD, axis=0)
        SR = np.stack(SR, axis=0)
        SIV = np.stack(SIV, axis=0)
        H_IF = np.stack(H_IF, axis=0)
        O_IF = np.stack(O_IF, axis=0)

        res_dt = {
            'PD': PD,
            'CP': CP,
            'SD': SD,
            'SR': SR,
            'SIV': SIV,
            'H_IF': H_IF,
            'O_IF': O_IF,
        }
        return res_dt
    
    def eval_using_simulator(self, data):
        #* align with 2023-NIPS-DeepSimHO
        PD, CP, SD, SR = [], [], [], []
        pd_hand_vert = data['pd_hand_vert']
        gt_hand_vert = data['gt_hand_vert']
        pd_obj_rt = data['pd_obj_rt']
        gt_obj_rt = data['gt_obj_rt']
        pd_force = data['force_global']
        gravity = data['gravity']
        obj_name = data['obj_name']

        pd_obj_vert_ls, gt_obj_vert_ls, obj_face_ls, siv_ls, h_if_ls, o_if_ls = [], [], [], [], [], []
        for i, obj_n in enumerate(obj_name):
            ori_obj_vert = self.obj_mesh[obj_n]['verts']
            obj_face = self.obj_mesh[obj_n]['faces']
            pd_obj_vert = ori_obj_vert @ pd_obj_rt[i, :, :3].T + pd_obj_rt[i, :, 3]
            gt_obj_vert = ori_obj_vert @ gt_obj_rt[i, :, :3].T + gt_obj_rt[i, :, 3]
            pd_obj_vert_ls.append(pd_obj_vert)
            gt_obj_vert_ls.append(gt_obj_vert)
            obj_face_ls.append(obj_face)
    
            siv = self.criterion_SIV(pd_hand_vert[i], pd_obj_rt[i], obj_name[i])
            siv_ls.append(siv)

            h_if, o_if = self.criterion_IF(pd_hand_vert[i], pd_obj_vert, gt_hand_vert[i], gt_obj_vert)
            h_if_ls.append(h_if)
            o_if_ls.append(o_if)

        if cfg.eval_with_simulator:
            pd, cp, sd, sr = self.criterion_2023_NIPS_DeepSimHO(pd_hand_vert, 
                                                                pd_obj_vert_ls, 
                                                                pd_obj_rt[:, :, :3], 
                                                                pd_obj_rt[:, :, 3], 
                                                                self.hand_faces, 
                                                                obj_face_ls, 
                                                                obj_name)
        else:
            pd, cp, sd, sr = np.zeros(len(obj_name)), np.zeros(len(obj_name)), np.zeros(len(obj_name)), np.zeros(len(obj_name))

        siv_ls = np.stack(siv_ls, axis=0)
        h_if_ls = np.stack(h_if_ls, axis=0)
        o_if_ls = np.stack(o_if_ls, axis=0)

        res_dt = {
            'PD': pd,
            'CP': cp,
            'SD': sd,
            'SR': sr,
            'SIV': siv_ls,
            'H_IF': h_if_ls,
            'O_IF': o_if_ls,
        }
        return res_dt            

    def __call__(self, data):
        """ data: {
            'pd_hand_vert': (N, V, 3),
            'gt_hand_vert': (N, V, 3),
            'pd_obj_rt': (N, 3, 4),
            'gt_obj_rt': (N, 3, 4),
            'pd_force_local': (N, 32, 3),
            'gravity': (N, 1, 3),
            'obj_name': (N, ),
        """
        data['force_point'], data['force_global'] = from_local_to_global(data['pd_force_local'], data['pd_hand_vert'])

        res_dt = {
            # 'Force': self.eval_use_force(data),
            'Simulator': self.eval_using_simulator(data),
        }
        return res_dt

    def criterion_SD_SR(self, pd_force, gravity, pd_obj_trans, gt_obj_trans):
        # TODO: implement using MOJOCO
        """ SD: simulation distance
            SR: success rate

            pd_force: (32, 3)
            gravity: (1, 3)
            pd_obj_trans: (3,)
            gt_obj_trans: (3,)
        """
        if use_gt := False:
            obj_trans = gt_obj_trans
        else:
            obj_trans = pd_obj_trans

        ext_force = np.concatenate([pd_force, gravity], axis=0).sum(axis=0) * self.g
        sd = 0.5 * ext_force * self.SD_time ** 2 # x = 0.5at^2
        sr = sd < self.SD_thresh
        return sd, sr
    

    def criterion_PD_CP(self, pd_hand_verts, pd_obj_verts, hand_faces, obj_faces):
        """ PD: penetration depth
            CP: contact percentage

            pd_hand_verts: (V1, 3)
            pd_obj_verts: (V2, 3)
            hand_faces: (F1, 3)
            obj_faces: (F2, 3)
        """
        pd_hand_verts = torch.from_numpy(pd_hand_verts).cuda()[None]
        pd_obj_verts = torch.from_numpy(pd_obj_verts).float().cuda()[None]
        hand_faces = hand_faces.cuda()[None]
        obj_faces = torch.from_numpy(obj_faces).cuda()[None]

        hand_to_obj_sdf = calculate_sdf(pd_hand_verts, 
                                        pd_obj_verts, 
                                        hand_faces, 
                                        obj_faces)[0, :778]
        pd = hand_to_obj_sdf.min().cpu().numpy()
        cp = pd < 0
        return pd, cp
    
    def criterion_2023_NIPS_DeepSimHO(self, pd_hand_verts, pd_obj_verts, pd_obj_rot, pd_obj_trans, hand_faces, obj_faces, obj_name):
        """ PD: penetration depth
            CP: contact percentage

            pd_hand_verts: (V1, 3)
            pd_obj_verts: (V2, 3)
            pd_obj_rot: (3, 3)
            pd_obj_trans: (3,)
            obj_faces: (F2, 3)
        """
        warnings.warn("The result of this function is not accurate, please refer to the official codes of 2023-NIPS-DeepSimHO.")

        pd_hand_verts = torch.from_numpy(pd_hand_verts)
        pd_obj_rot = torch.from_numpy(pd_obj_rot).float()
        pd_obj_rot = matrix_to_quaternion(pd_obj_rot)
        pd_obj_trans = torch.from_numpy(pd_obj_trans).float()

        camera_dir = 'camera' if cfg.dataset_name == 'dexycb' else 'OpenGL'
        final_state = MuJoCoMeshSimulatorFast.batched_simulate(obj_name, pd_hand_verts, pd_obj_rot, pd_obj_trans, camera_dir)

        sd = torch.norm(final_state[:,:3] - pd_obj_trans.cuda(), p=2, dim=-1) # (B, 1)
        sr = (sd <= self.SD_thresh).int() # (B, 1)

        b_cn, _ = MuJoCoMeshSimulatorFast.batched_get_contact(obj_name, pd_hand_verts, pd_obj_rot, pd_obj_trans, camera_dir)
        b_cp = (b_cn > 0).int()
        
        hand_to_obj_sdf = []
        for i in range(len(pd_hand_verts)):
            hand_vert_i = pd_hand_verts[i].cuda()[None]
            obj_verts_i = torch.from_numpy(pd_obj_verts[i]).float().cuda()[None]
            hand_faces_i = hand_faces.cuda()[None]
            obj_faces_i = torch.from_numpy(obj_faces[i]).cuda()[None]
            x = calculate_sdf(hand_vert_i, obj_verts_i, hand_faces_i, obj_faces_i)[:, :778]
            hand_to_obj_sdf.append(x)

        b_pd = []
        for k in hand_to_obj_sdf:
            b_pd.append(torch.abs(k[k < 0].min()))
        b_pd = torch.stack(b_pd).cuda()

        sd = sd.cpu().numpy()
        sr = sr.cpu().numpy()
        cp = b_cp.cpu().numpy()
        pd = b_pd.cpu().numpy()
        return pd, cp, sd, sr
    
    
    def criterion_SIV(self, hand_verts, obj_rt, obj_name):
        """ SIV: solid intersection volume
            align with 2021-CVPR-CPF

            pd_hand_verts: (V1, 3)
            pd_obj_verts: (3, 4)
            obj_name: str
        """
        obj_vox_can_np = self.obj_vox[obj_name]['points']
        obj_vox_el_vol = self.obj_vox[obj_name]['element_volume']
        obj_r = obj_rt[:, :3]
        obj_t = obj_rt[:, 3]

        siv, _, _ = solid_intersection_volume(
            hand_verts,
            self.hand_close_faces_np,
            obj_vox_can_np,
            obj_t,
            obj_r,
            obj_vox_el_vol
        )
        # # region [test]
        # obj_vox_abs = obj_vox_can_np @ obj_r.T + obj_t
        # save_dt = {
        #     'hand_verts_#FF0000': hand_verts,
        #     'obj_vox_abs_#00FF00': obj_vox_abs,
        # }
        # with open('tmp.pkl', 'wb') as f:
        #     pickle.dump(save_dt, f)
        # print(siv)
        # exit()
        # # endregion
        return siv


    def criterion_IF(self, pd_hand_verts, pd_obj_verts, gt_hand_verts, gt_obj_verts):
        """ PD: penetration depth
            CP: contact percentage

            pd_hand_verts: (V1, 3)
            pd_obj_verts: (V2, 3)
            gt_hand_verts: (V1, 3)
            gt_obj_verts: (V2, 3)
            hand_faces: (F1, 3)
            obj_faces: (F2, 3)
            contact_thresh: float
        """
        pd_hand_verts = torch.from_numpy(pd_hand_verts).float().cuda()
        pd_obj_verts = torch.from_numpy(pd_obj_verts).float().cuda()
        gt_hand_verts = torch.from_numpy(gt_hand_verts).float().cuda()
        gt_obj_verts = torch.from_numpy(gt_obj_verts).float().cuda()

        pd_dist_mat = torch.cdist(pd_hand_verts, pd_obj_verts)
        gt_dist_mat = torch.cdist(gt_hand_verts, gt_obj_verts)

        pd_h2o = pd_dist_mat.min(dim=1)[0]
        pd_o2h = pd_dist_mat.min(dim=0)[0]
        gt_h2o = gt_dist_mat.min(dim=1)[0]
        gt_o2h = gt_dist_mat.min(dim=0)[0]
        
        h_iF = torch.abs(pd_h2o - gt_h2o).mean().cpu().numpy()
        o_iF = torch.abs(pd_o2h - gt_o2h).mean().cpu().numpy()

        return h_iF, o_iF


    def postprocess(self, dt_ls):
        force_dt = defaultdict(list)
        simulator_dt = defaultdict(list)

        for dt in dt_ls:
            # for k, v in dt['Force'].items():
            #     force_dt[k].append(v)
            for k, v in dt['Simulator'].items():
                simulator_dt[k].append(v)

        # for k, v in force_dt.items():
        #     force_dt[k] = np.concatenate(v, axis=0).mean()
        #     if k in ['PD', 'CP', 'SD', 'SR']:
        #         force_dt[k] = np.around(force_dt[k]*100, 2)
        #     if k in ['SIV']:
        #         force_dt[k] = np.around(force_dt[k]*1000000, 2)
        #     if k in ['H_IF', 'O_IF']:
        #         force_dt[k] = np.around(force_dt[k]*1000, 2)

        for k, v in simulator_dt.items():
            simulator_dt[k] = np.concatenate(v, axis=0).mean()
            if k in ['PD', 'CP', 'SD', 'SR']:
                simulator_dt[k] = np.around(simulator_dt[k]*100, 2)
            if k in ['SIV']:
                simulator_dt[k] = np.around(simulator_dt[k]*1000000, 2) # cm^3
            if k in ['H_IF', 'O_IF']:
                simulator_dt[k] = np.around(simulator_dt[k]*1000, 2)

        res_dt = {
            # 'Force': force_dt,
            'Simulator': simulator_dt,
        }

        df = pandas.DataFrame(res_dt).T
        return df
        


    

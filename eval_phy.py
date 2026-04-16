import os
import pickle
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from lib.configs.args import cfg
from lib.dataset.dexycb6 import DexYCBDataset_Force
from lib.dataset.ho3d_multi_bbx import HO3DDataset_Train, HO3DDataset_Valid, HO3DDataset_Test
from lib.engine.test_physics import TesterPhysics
from lib.utils.misc_fn import to_numpy

def obj_9D_to_mat(obj_9D):
    assert isinstance(obj_9D, torch.Tensor)
    obj_rot6d = obj_9D[..., :6]
    obj_rotmat = rotation_6d_to_matrix(obj_rot6d)
    obj_rt = torch.cat([obj_rotmat, obj_9D[..., 6:9, None]], dim=-1)
    return obj_rt

def __postprocess_obj_rt(rot6d, root_joint):
    rt = obj_9D_to_mat(rot6d)
    rt[..., 3] = torch.einsum("b...i,bi->b...i", torch.ones_like(rt[..., 3]), root_joint) + rt[..., 3]
    return rt

def __postprocess_hand_vert(vert, root_joint, is_right):
    flipped_idx = torch.arange(vert.shape[0], device=is_right.device)[~is_right]
    vert[flipped_idx, ..., 0] = -vert[flipped_idx, ..., 0]
    vert = torch.einsum("b...i,bi->b...i", torch.ones_like(vert), root_joint) + vert
    return vert

def postprocess(data, root_joint, is_right=None):
    for k in list(data.keys()):
        if k in ['obj_inprocess', 'obj_final', 'gt_obj', 'obj_mean']:
            data[f'{k}_rt'] = __postprocess_obj_rt(data[k], root_joint)

    for k, v in data.items():
        if k in ['hand_vert', 'hand_joint', 'final_hand_vert', 'final_hand_joint', 'mean_hand_vert', 'mean_hand_joint']:
            data[k] = __postprocess_hand_vert(v, root_joint, is_right)
        elif k in ['inprocess_hand_vert', 'inprocess_hand_joint']:
            _is_right = torch.zeros(v.shape[0], dtype=torch.bool, device=v.device) + is_right[0]
            _root_joint = torch.zeros([v.shape[0], root_joint.shape[-1]], device=v.device) + root_joint[0]
            data[k] = __postprocess_hand_vert(v, _root_joint, _is_right)
    return data


def eval(pd_res_dt, test_loader):
    tester_physics = TesterPhysics()
    collector_physics = []
    for i, batch in enumerate(tqdm.tqdm(test_loader)):
        batch = postprocess(batch, batch['root_joint'])
        batch = to_numpy(batch)

        hand_vert_ls = []
        obj_rt_ls = []

        for p in batch['rgb_path']:
            k = os.path.join(*p.split('/')[-4:])
            assert k in pd_res_dt, f"{k} not in prediction result"

            hand_vert_ls.append(pd_res_dt[k]['hand_vert'])
            obj_rt_ls.append(pd_res_dt[k]['obj_rt'])
        hand_vert_ls = np.stack(hand_vert_ls, axis=0)
        obj_rt_ls = np.stack(obj_rt_ls, axis=0)

        res_dt = {
            'mean_hand_vert': hand_vert_ls,
            'obj_mean_rt': obj_rt_ls,
            'force_local': np.zeros([hand_vert_ls.shape[0], 32, 3]),
        }

        eval_phy_dt = {
            'pd_hand_vert': res_dt['mean_hand_vert'],
            'gt_hand_vert': batch['gt_hand_vert'],
            'pd_obj_rt': res_dt['obj_mean_rt'],
            'gt_obj_rt': batch['gt_obj_rt'],
            'pd_force_local': res_dt['force_local'],
            'gravity': batch['gravity'],
            'obj_name': batch['obj_name'],
        }
        collector_physics.append(tester_physics(eval_phy_dt))
    
    phy_info = tester_physics.postprocess(collector_physics)
    info = f"Physics Evaluation: \n"
    info += f"{phy_info}"
    print(info)


def load_prediction_result(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    res = {}
    for d in data:
        obj = d["pd_obj_rt"]
        rgb_path = d["path"]
        hand_vert = d["pd_hand_vert"].astype(np.float32)
        hand_joint = d["pd_hand_joint"]

        for i, _ in enumerate(rgb_path):
            p = rgb_path[i].split("/")
            p = os.path.join(*p[-4:])
            res[p] = {
                "obj_rt": obj[i],
                "hand_vert": hand_vert[i],
                "hand_joint": hand_joint[i]
            }
    return res


if __name__ == "__main__":
    path = "results/dexycb-physics/2022-CVPR-Artiboost_DexYCB-physics-result.pkl"
    res = load_prediction_result(path)
    testset = DexYCBDataset_Force(
        cfg.data_dir,
        is_train=False,
        aug=cfg,
        cfg=cfg,
    )
    testing_dataloader = DataLoader(
        testset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.eval_num_workers,
        pin_memory=False,
        drop_last=False,
    )
    eval(res, testing_dataloader)

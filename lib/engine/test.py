import os
import numpy as np
import pickle
import time
import torch
import tqdm
import json
import math

from lib.configs.args import cfg
from lib.dataset.base import YCB_MESHES, YCB_CLASSES, YCB_ID
from lib.utils.transform_fn import rigid_align_AtoB

# from 2022-CVPR-ArtiBoost
def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
    
# from 2022-CVPR-ArtiBoost
def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        [[0.0, -direction[2], direction[1]], [direction[2], 0.0, -direction[0]], [-direction[1], direction[0], 0.0]]
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def get_symmetry_transformations(model_info, max_sym_disc_step):
    """Returns a set of symmetry transformations for an object model.

    :param model_info: See files models_info.json provided with the datasets.
    :param max_sym_disc_step: The maximum fraction of the object diameter which
      the vertex that is the furthest from the axis of continuous rotational
      symmetry travels between consecutive discretized rotations.
    :return: The set of symmetry transformations.
    """
    # Discrete symmetries.
    trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]  # Identity.
    if "symmetries_discrete" in model_info:
        for sym in model_info["symmetries_discrete"]:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_disc.append({"R": R, "t": t})

    # Discretized continuous symmetries.
    trans_cont = []
    if "symmetries_continuous" in model_info:
        for sym in model_info["symmetries_continuous"]:
            axis = np.array(sym["axis"])
            offset = np.array(sym["offset"]).reshape((3, 1))

            # (PI * diam.) / (max_sym_disc_step * diam.) = discrete_steps_count
            discrete_steps_count = int(np.ceil(np.pi / max_sym_disc_step))

            # Discrete step in radians.
            discrete_step = 2.0 * np.pi / discrete_steps_count

            for i in range(1, discrete_steps_count):
                R = rotation_matrix(i * discrete_step, axis)[:3, :3]
                t = -R.dot(offset) + offset
                trans_cont.append({"R": R, "t": t})

    # Combine the discrete and the discretized continuous symmetries.
    trans = []
    for tran_disc in trans_disc:
        if len(trans_cont):
            for tran_cont in trans_cont:
                R = tran_cont["R"].dot(tran_disc["R"])
                t = tran_cont["R"].dot(tran_disc["t"]) + tran_cont["t"]
                trans.append({"R": R, "t": t})
        else:
            trans.append(tran_disc)

    return trans


# from 2024-CVPR-HOISDF
@torch.no_grad()
def compute_obj_metrics_dexycb(pred_meshes, target_meshes):
    B, N, _ = pred_meshes.shape
    add_gt = target_meshes.unsqueeze(1).repeat(1, N, 1, 1)
    add_pred = pred_meshes.unsqueeze(2).repeat(1, 1, N, 1)
    dis = torch.norm(add_gt - add_pred, dim=-1)
    add_bias = torch.mean(torch.min(dis, dim=2)[0], dim=1)
    add_bias = add_bias.detach().cpu()

    corner_indexes = torch.tensor(
        [[0, 1, 0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 1], [0, 0, 0, 1, 0, 1, 1, 1]]
    ).cuda()
    target_mm = torch.stack(
        [torch.min(target_meshes, dim=1)[0], torch.max(target_meshes, dim=1)[0]], dim=2
    )
    target_bboxes = torch.stack(
        [
            target_mm[:, 0, corner_indexes[0]],
            target_mm[:, 1, corner_indexes[1]],
            target_mm[:, 2, corner_indexes[2]],
        ],
        dim=2,
    )
    pred_mm = torch.stack(
        [torch.min(pred_meshes, dim=1)[0], torch.max(pred_meshes, dim=1)[0]], dim=2
    )
    pred_bboxes = torch.stack(
        [
            pred_mm[:, 0, corner_indexes[0]],
            pred_mm[:, 1, corner_indexes[1]],
            pred_mm[:, 2, corner_indexes[2]],
        ],
        dim=2,
    )

    MCE_error = (
        (pred_bboxes - target_bboxes.float()).norm(2, -1).mean(-1).detach().cpu()
    )

    return add_bias, MCE_error


class TesterObject:
    def __init__(self):
        self.cfg = cfg
        # if self.cfg.dataset_name == 'dexycb':
        self.obj_mesh = YCB_MESHES

        # region [sym corner]
        # modified from 2022_CVPR_ArtiBoost
        self.model_info = json.load(open("asset/2023_NIPS_DeepSimHO/assets_models_info.json", "r"))
        self.max_sym_disc_step = 0.01

        self.model_sym = {}
        max_sym_len = 0
        for obj_idx in range(1, len(self.model_info) + 1):
            self.model_sym[obj_idx] = get_symmetry_transformations(self.model_info[str(obj_idx)], self.max_sym_disc_step)
            max_sym_len = max(max_sym_len, len(self.model_sym[obj_idx]))
        R, t = [], []
        for obj_idx in range(1, len(self.model_info) + 1):
            obj_R, obj_t = [], []
            for transf in self.model_sym[obj_idx]:
                obj_R.append(transf["R"])
                obj_t.append(transf["t"])
            while len(obj_R) < max_sym_len:
                obj_R.append(np.eye(3))
                obj_t.append(np.zeros((3, 1)))
            obj_R = np.stack(obj_R)  # [max_sym_len, 3, 3]
            obj_t = np.stack(obj_t)  # [max_sym_len, 3, 1]
            R.append(obj_R)
            t.append(obj_t / 1000.0)  # mm to m
        self.R = np.stack(R)  # [N, max_sym_len, 3, 3]
        self.t = np.stack(t)  # [N, max_sym_len, 3, 1]
        # endregion

    def __call__(self, data):
        """ data: {
            'pd_rt': (N, ..., 3, 4),
            'gt_rt': (N, 3, 4),
            'obj_name': (N, ),
            'cam_intr': (N, 3, 3),
        }
        """
        if isinstance(data, str):
            with open(data, 'rb') as f:
                data = pickle.load(f)

        length = data['gt_rt'].shape[0]
        MCE, MCE2, SMCE, OCE, ADD, ADDS, ADD01d, ADDS01d, REP, REP5 = [], [], [], [], [], [], [], [], [], []

        # ===== 新增：F-score（多阈值）与 Chamfer 距离收集容器 =====
        FSCORE_KEYS = ["FSCORE@2mm", "FSCORE@5mm", "FSCORE@10mm", "FSCORE@2cm", "FSCORE@5cm", "FSCORE@10cm"]
        FSCORE = {k: [] for k in FSCORE_KEYS}
        CD = []
        # ======================================================

        for i in range(length):
            pd_rt = data['pd_rt'][i]
            gt_rt = data['gt_rt'][i]
            obj_name = data['obj_name'][i]
            cam_intr = data['cam_intr'][i]

            mce, oce = self.criterion_MCE_OCE(pd_rt, gt_rt, obj_name)
            mce2 = self.criterion_MCE2(pd_rt, gt_rt, obj_name)
            # smce = self.criterion_SMCE(pd_rt, gt_rt, obj_name) # very slow
            add, adds, rep = self.criterion_ADD_REP(pd_rt, gt_rt, obj_name, cam_intr)
            add01d, adds01d = self.cal_ADD01d(add, adds, obj_name)
            rep5 = self.cal_REP5(rep)

            # ===== 新增：F-score 与 Chamfer 距离 =====
            fscore_dict, cd = self.criterion_FSCORE(pd_rt, gt_rt, obj_name)
            for k in FSCORE_KEYS:
                FSCORE[k].append(fscore_dict[k])
            CD.append(cd)
            # ======================================

            MCE.append(mce)
            MCE2.append(mce2)
            # SMCE.append(smce)
            OCE.append(oce)
            ADD.append(add)
            ADDS.append(adds)
            ADD01d.append(add01d)
            ADDS01d.append(adds01d)
            REP.append(rep)
            REP5.append(rep5)

        MCE = np.stack(MCE, axis=0)
        MCE2 = np.stack(MCE2, axis=0)
        # SMCE = np.stack(SMCE, axis=0)
        OCE = np.stack(OCE, axis=0)
        ADD = np.stack(ADD, axis=0)
        ADDS = np.stack(ADDS, axis=0)
        ADD01d = np.stack(ADD01d, axis=0)
        ADDS01d = np.stack(ADDS01d, axis=0)
        REP = np.stack(REP, axis=0)
        REP5 = np.stack(REP5, axis=0)

        # 新增：堆叠 FSCORE 与 CD
        CD = np.stack(CD, axis=0)
        for k in FSCORE_KEYS:
            FSCORE[k] = np.stack(FSCORE[k], axis=0)

        MCE_dt, MCE2_dt, SMCE_dt, OCE_dt, ADD_dt, ADDS_dt, ADD01d_dt, ADDS01d_dt, REP_dt, REP5_dt = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        # 新增
        FSCORE_dt = {k: {} for k in FSCORE_KEYS}
        CD_dt = {}

        for k in YCB_MESHES.keys():
            if k == '051_large_clamp':
                continue
            sel = data['obj_name'] == k
            MCE_dt[k] = MCE[sel]
            MCE2_dt[k] = MCE2[sel]
            # SMCE_dt[k] = SMCE[sel]
            OCE_dt[k] = OCE[sel]
            ADD_dt[k] = ADD[sel]
            ADDS_dt[k] = ADDS[sel]
            ADD01d_dt[k] = ADD01d[sel]
            ADDS01d_dt[k] = ADDS01d[sel]
            REP_dt[k] = REP[sel]
            REP5_dt[k] = REP5[sel]
            # 新增
            for kk in FSCORE_KEYS:
                FSCORE_dt[kk][k] = FSCORE[kk][sel]
            CD_dt[k] = CD[sel]

        # 注意：average_instance 放在循环外，避免覆盖
        MCE_dt['average_instance'] = MCE
        MCE2_dt['average_instance'] = MCE2
        # SMCE_dt['average_instance'] = SMCE
        OCE_dt['average_instance'] = OCE
        ADD_dt['average_instance'] = ADD
        ADDS_dt['average_instance'] = ADDS
        ADD01d_dt['average_instance'] = ADD01d
        ADDS01d_dt['average_instance'] = ADDS01d
        REP_dt['average_instance'] = REP
        REP5_dt['average_instance'] = REP5
        for kk in FSCORE_KEYS:
            FSCORE_dt[kk]['average_instance'] = FSCORE[kk]
        CD_dt['average_instance'] = CD

        res_dt = {
            'MCE': MCE_dt,
            'MCE2': MCE2_dt,
            # 'SMCE': SMCE_dt,
            'OCE': OCE_dt,
            'ADD': ADD_dt,
            'ADDS': ADDS_dt,
            'ADD01d': ADD01d_dt,
            'ADDS01d': ADDS01d_dt,
            'REP': REP_dt,
            # 新增
            'CD': CD_dt,  # Chamfer-L2 (m)
        }
        # 把各个 FSCORE@* 合并进入结果
        res_dt.update({kk: FSCORE_dt[kk] for kk in FSCORE_KEYS})

        return res_dt

    def criterion_MCE_OCE(self, pd_rt, gt_rt, obj_name):
        """
            Args:
                pd_rt: (..., 3, 4)
                gt_rt: (3, 4)
            Return:
                mce: (..., )
                oce: (..., )
        """
        obj_bbox3d = self.obj_mesh[obj_name]['bbox3d']

        pd_bbox3d = np.einsum("ni,...ij->...nj", obj_bbox3d, pd_rt[..., :3].swapaxes(-1, -2)) + pd_rt[..., 3][..., None, :]
        gt_bbox3d = np.einsum("ni,ij->nj", obj_bbox3d, gt_rt[:, :3].swapaxes(-1, -2)) + gt_rt[:, 3][None, :]

        mce = np.linalg.norm(pd_bbox3d - gt_bbox3d, axis=-1).mean(-1)
        
        pd_center = pd_bbox3d.mean(-2)
        gt_center = gt_bbox3d.mean(-2)
        oce = np.linalg.norm(pd_center - gt_center, axis=-1)
        
        return mce, oce
    
    #* modified from 2022_CVPR_ArtiBoost
    def criterion_SMCE(self, pd_rt, gt_rt, obj_name):
        """
            Args:
                pd_rt: (..., 3, 4)
                gt_rt: (3, 4)
            Return:
                mce: (..., )
                oce: (..., )
        """
        ori_bbox3d = self.obj_mesh[obj_name]['bbox3d']
        sym_R = self.R[YCB_ID[obj_name] - 1]
        sym_t = self.t[YCB_ID[obj_name] - 1]

        sym_bbox3d = np.einsum("ni,kji->knj", ori_bbox3d, sym_R) + sym_t.swapaxes(-1, -2)
        gt_sym_bbox3d = np.einsum("kni,ij->knj", sym_bbox3d, gt_rt[:, :3].swapaxes(-1, -2)) + gt_rt[:, 3][None, None, :]

        pd_bbox3d = np.einsum("ni,...ij->...nj", ori_bbox3d, pd_rt[..., :3].swapaxes(-1, -2)) + pd_rt[..., 3][..., None, :]

        #! note: in 2022_CVPR_ArtiBoost, the invisible corners are masked out
        smce = pd_bbox3d[..., None, :, :] - gt_sym_bbox3d
        smce = np.linalg.norm(smce, axis=-1).mean(-1).min(-1)
        return smce
    
    #* Align with 2024-CVPR-HOISDF
    def criterion_MCE2(self, pd_rt, gt_rt, obj_name):
        """ pd_rt: (..., 3, 4)
            gt_rt: (3, 4)
        """
        if use_sampled_vertices := True:
            obj_vert = self.obj_mesh[obj_name]['verts_sampled']
        else:
            obj_vert = self.obj_mesh[obj_name]['verts']

        pd_verts = np.einsum("ni,...ij->...nj", obj_vert, pd_rt[..., :3].swapaxes(-1, -2)) + pd_rt[..., 3][..., None, :]
        gt_verts = np.einsum("ni,ij->nj", obj_vert, gt_rt[:, :3].swapaxes(-1, -2)) + gt_rt[:, 3][None, :]

        pd_verts_pt = torch.from_numpy(pd_verts).float().cuda()[None]
        gt_verts_pt = torch.from_numpy(gt_verts).float().cuda()[None]
        
        _, mce = compute_obj_metrics_dexycb(pd_verts_pt, gt_verts_pt)
        return mce[0].numpy()

    @torch.no_grad()
    def criterion_ADD_REP(self, pd_rt, gt_rt, obj_name, cam_intr):
        """
            Args:
                pd_rt: (..., 3, 4)
                gt_rt: (3, 4)
            Return:
                add: (..., )
                adds: (..., )
                rep: (..., )
        """
        if use_sampled_vertices := True:
            obj_vert = self.obj_mesh[obj_name]['verts_sampled']
        else:
            obj_vert = self.obj_mesh[obj_name]['verts']

        pd_verts = np.einsum("ni,...ij->...nj", obj_vert, pd_rt[..., :3].swapaxes(-1, -2)) + pd_rt[..., 3][..., None, :]
        gt_verts = np.einsum("ni,ij->nj", obj_vert, gt_rt[:, :3].swapaxes(-1, -2)) + gt_rt[:, 3][None, :]

        pd_verts_pt = torch.from_numpy(pd_verts).float().cuda()
        gt_verts_pt = torch.from_numpy(gt_verts).float().cuda()

        adds = torch.cdist(pd_verts_pt, gt_verts_pt).min(-1)[0].mean(-1).cpu().numpy() # checked
        add = torch.norm(pd_verts_pt - gt_verts_pt, dim=-1).mean(-1).cpu().numpy()

        pd_vert_proj = np.einsum("...ni,ij->...nj", pd_verts, cam_intr.swapaxes(-1, -2)) / (pd_verts[..., 2:3] + 1e-7)
        pd_vert_proj = pd_vert_proj[..., :2]
        gt_vert_proj = np.einsum("ni,ij->nj", gt_verts, cam_intr.swapaxes(-1, -2)) / (gt_verts[..., 2:3] + 1e-7)
        gt_vert_proj = gt_vert_proj[..., :2]
        rep = np.linalg.norm(pd_vert_proj - gt_vert_proj, axis=-1).mean(-1)

        return add, adds, rep

    @torch.no_grad()
    def criterion_FSCORE(self, pd_rt, gt_rt, obj_name, th_list=None):
        """
        Args:
            pd_rt: (..., 3, 4)
            gt_rt: (3, 4)
            th_list: list[float] in meters (default: [2mm, 5mm, 10mm, 2cm, 5cm, 10cm])
        Returns:
            fscore_dict: { 'FSCORE@2mm': (...,), 'FSCORE@5mm': (...,), ... }
            cd: (...,)  # Chamfer-L2 (non-squared), meters
        """
        if th_list is None:
            th_list = [0.002, 0.005, 0.010, 0.020, 0.050, 0.100]

        if use_sampled_vertices := False:
            obj_vert = self.obj_mesh[obj_name]['verts_sampled']
        else:
            obj_vert = self.obj_mesh[obj_name]['verts']

        pd_verts = np.einsum("ni,...ij->...nj", obj_vert, pd_rt[..., :3].swapaxes(-1, -2)) + pd_rt[..., 3][..., None, :]
        gt_verts = np.einsum("ni,ij->nj", obj_vert, gt_rt[:, :3].swapaxes(-1, -2)) + gt_rt[:, 3][None, :]

        pd_verts_pt = torch.from_numpy(pd_verts).float().cuda()  # (..., P, 3)
        gt_verts_pt = torch.from_numpy(gt_verts).float().cuda()  # (Q, 3)

        if pd_verts_pt.ndim == 2:
            pd_verts_pt = pd_verts_pt.unsqueeze(0)  # (1, P, 3)
        gt_verts_pt = gt_verts_pt.unsqueeze(0).expand(pd_verts_pt.size(0), -1, -1)  # (B, Q, 3)

        dmat = torch.cdist(pd_verts_pt, gt_verts_pt)  # (B, P, Q)
        d_pred2gt, _ = dmat.min(dim=2)  # (B, P)
        d_gt2pred, _ = dmat.min(dim=1)  # (B, Q)

        # Chamfer-L2 (non-squared)
        cd = 0.5 * (d_pred2gt.mean(dim=1) + d_gt2pred.mean(dim=1))  # (B,)

        eps = 1e-6
        fscore_dict = {}
        for th in th_list:
            precision = (d_pred2gt < th).float().mean(dim=1)
            recall    = (d_gt2pred < th).float().mean(dim=1)
            fscore    = (2 * precision * recall) / (precision + recall + eps)

            # 阈值标签：<= 10mm 都用 mm，其他用 cm；用 round 防止 0.0100000002 这类浮点误差
            if th <= 0.010 + 1e-9:
                tag = f"FSCORE@{int(round(th * 1000))}mm"   # 2/5/10 mm -> 2/5/10
            else:
                tag = f"FSCORE@{int(round(th * 100))}cm"    # 2/5/10 cm -> 2/5/10
            fscore_dict[tag] = fscore.detach().cpu().numpy()


        return fscore_dict, cd.detach().cpu().numpy()
    
    def cal_ADD01d(self, add, adds, obj_name):
        """
            Args:
                add: (..., )
                adds: (..., )
            Return:
                add01d: (..., )
        """
        diameter = self.obj_mesh[obj_name]['diameter']

        add01d = add <= diameter * 0.1
        adds01d = adds <= diameter * 0.1
        return add01d, adds01d

    def cal_REP5(self, rep):
        return rep < 5
    
    def postprocess(self, res_dt):
        is_multihyper = len(next(iter(res_dt['ADDS'].values())).shape) > 1

        post_dt = {}
        for k, v in res_dt.items():
            tmp_dt = {}
            class_vals = []

            for kk, vv in v.items():
                if vv.shape[0] == 0:
                    continue

                if not is_multihyper:
                    val = vv.mean()
                    stacked_for_avg = vv
                else:
                    # 多超参：选择更优的那一条
                    if k in ['MCE', 'MCE2', 'SMCE', 'OCE', 'ADD', 'ADDS', 'CD', 'REP']:
                        best = vv.min(-1)        # 距离/误差类 越小越好
                        val = best.mean(-1)
                        stacked_for_avg = best
                    elif k in ['ADD01d', 'ADDS01d', 'REP5'] or k.startswith('FSCORE@'):
                        best = vv.max(-1)        # 准确率/命中率类 越大越好
                        val = best.mean(-1)
                        stacked_for_avg = best
                    else:
                        val = vv.mean(-1)
                        stacked_for_avg = vv

                tmp_dt[kk] = val
                if kk != 'average_instance':
                    class_vals.append(stacked_for_avg)

            # 类均值（各类别/实例的聚合）
            if len(class_vals) > 0:
                concat_for_class = np.concatenate(class_vals)
                tmp_dt['average_class'] = concat_for_class.mean()
                tmp_dt['average_instance'] = concat_for_class.mean()
            else:
                tmp_dt['average_class'] = np.nan
                tmp_dt['average_instance'] = np.nan

            post_dt[k] = tmp_dt

        out_dt = self.format(post_dt)
        return out_dt
    
    def format(self, post_dt):
        for k, v in post_dt.items():
            for kk, vv in v.items():
                if k in ['MCE', 'MCE2', 'SMCE', 'OCE', 'ADD', 'ADDS', 'CD']:
                    # 距离（m -> mm），保留 2 位小数（向下截断）
                    v[kk] = int(vv*100000) / 100 if np.isfinite(vv) else vv # to mm
                elif k in ['ADD01d', 'ADDS01d', 'REP5'] or k.startswith('FSCORE@'):
                    # 百分比，保留 2 位小数（向下截断）
                    v[kk] = int(vv*10000) / 100 if np.isfinite(vv) else vv # to percent
                elif k in ['REP']:
                    # 像素
                    v[kk] = int(vv*100) / 100 if np.isfinite(vv) else vv # to pixel
                
        return post_dt


class TesterHand:
    def __init__(self):
        self.cfg = cfg

    def __call__(self, data):
        """ data: {
            'is_right': (N, ),
            'gt_joint': (N, 21, 3),
            'pd_joint': (N, ..., 21, 3),
            'gt_vert': (N, 778, 3),
            'pd_vert': (N, ..., 778, 3),
        }
        """
        if isinstance(data, str):
            with open(data, 'rb') as f:
                data = pickle.load(f)

        length = data['gt_joint'].shape[0]
        IS_RIGHT, MJE, PA_MJE, JE, MVE, PAMVE = [], [], [], [], [], []
        for i in range(length):
            gt_joint = data['gt_joint'][i]
            pd_joint = data['pd_joint'][i]
            is_right = data['is_right'][i]
            gt_vert = data['gt_vert'][i]
            pd_vert = data['pd_vert'][i]

            mje, pa_mje, je = self.criterion_MJE_PAMJE(gt_joint, pd_joint)
            mve, pa_mve, _ = self.criterion_MJE_PAMJE(gt_vert, pd_vert)
            IS_RIGHT.append(is_right)
            MJE.append(mje)
            PA_MJE.append(pa_mje)
            JE.append(je)
            MVE.append(mve)
            PAMVE.append(pa_mve)

        IS_RIGHT = np.stack(IS_RIGHT, axis=0)
        MJE = np.stack(MJE, axis=0)
        PA_MJE = np.stack(PA_MJE, axis=0)
        JE = np.stack(JE, axis=0)
        MVE = np.stack(MVE, axis=0)
        PAMVE = np.stack(PAMVE, axis=0)

        MJE_dt, PA_MJE_dt, JE_dt, MVE_dt, PAMVE_dt = {}, {}, {}, {}, {}
        MJE_dt['right'] = MJE[IS_RIGHT]
        MJE_dt['left'] = MJE[~IS_RIGHT]
        MJE_dt['both'] = MJE
        PA_MJE_dt['right'] = PA_MJE[IS_RIGHT]
        PA_MJE_dt['left'] = PA_MJE[~IS_RIGHT]
        PA_MJE_dt['both'] = PA_MJE
        JE_dt['right'] = JE[IS_RIGHT]
        JE_dt['left'] = JE[~IS_RIGHT]
        JE_dt['both'] = JE
        MVE_dt['right'] = MVE[IS_RIGHT]
        MVE_dt['left'] = MVE[~IS_RIGHT]
        MVE_dt['both'] = MVE
        PAMVE_dt['right'] = PAMVE[IS_RIGHT]
        PAMVE_dt['left'] = PAMVE[~IS_RIGHT]
        PAMVE_dt['both'] = PAMVE

        JE_split_dt = {}
        for i in range(21):
            JE_split_dt[f"MJE_{i}"] = {k: v[..., i] for k, v in JE_dt.items()}

        res_dt = {
            'MJE': MJE_dt,
            'PA_MJE': PA_MJE_dt,
            'MVE': MVE_dt,
            'PAMVE': PAMVE_dt,
        }
        res_dt.update(JE_split_dt)
        return res_dt
    
    def criterion_MJE_PAMJE(self, gt_joint, pd_joint):
        """
            Args:
                gt_joint: (21, 3)
                pd_joint: (..., 21, 3)
            Return:
                mje: (..., )
                pa_mje: (..., )
                je: (..., 21)
        """
        je = np.linalg.norm(gt_joint - pd_joint, axis=-1)
        mje = je.mean(-1)
        
        if len(pd_joint.shape) == 2:
            pd_joint_aligned = rigid_align_AtoB(pd_joint, gt_joint)
        else:
            all_aligned = []
            for i in range(pd_joint.shape[0]):
                pd_joint_aligned_i = rigid_align_AtoB(pd_joint[i], gt_joint)
                all_aligned.append(pd_joint_aligned_i)
            pd_joint_aligned = np.stack(all_aligned, axis=0)
        pa_mje = np.linalg.norm(gt_joint - pd_joint_aligned, axis=-1).mean(-1)
        return mje, pa_mje, je
    

import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix

from lib.dataset.dexycb4 import DexYCBDataset, YCB_MESHES
from lib.dataset.ho3d2 import HO3DDataset_Train, HO3DDataset_Test
from lib.model.physics import ForceEncoder, from_local_to_global, HeadForce
from lib.engine.base_trainer import BaseTrainer
from lib.configs.args import Config
from lib.utils.misc_fn import to_device, to_numpy, to_tensor
from lib.utils.hand_fn import VERT2JOINT, SKELETON

class ForceOptimizer(BaseTrainer):
    def __init__(self, cfg:Config):
        cfg.model = ''
        cfg.mode = 'optim'
        cfg.bbox_scale_factor = 1.5
        self.num_anchor = 32
        super(ForceOptimizer, self).__init__(cfg)
        self.cfg = cfg
        self.get_obj_mesh()

    def get_model(self):
        self.model = HeadForce(1)
        self.model.to(self.device)

    def get_optimizer(self):
        self.scale = nn.Parameter(torch.ones([self.cfg.batch_size, self.num_anchor], device=self.device) * 0.05)
        self.weight = nn.Parameter(torch.zeros([self.cfg.batch_size, self.num_anchor, 8], device=self.device))
        self.optimizer1 = optim.AdamW([self.weight], betas=(0.9, 0.999), eps=1e-8, lr=1e-3)
        self.optimizer2 = optim.AdamW([self.scale, self.weight], betas=(0.9, 0.999), eps=1e-8, lr=1e-3)

    def get_scheduler(self):
        ...

    def init_param(self):
        self.scale.data = torch.ones([self.cfg.batch_size, self.num_anchor], device=self.device) * 0.05
        self.weight.data = torch.zeros([self.cfg.batch_size, self.num_anchor, 8], device=self.device)
    
    def get_dataloader(self):
        if self.cfg.dataset_name == 'dexycb':
            self.trainset = DexYCBDataset(
                self.cfg.data_dir,
                is_train=True,
                aug=self.cfg,
                cfg=self.cfg,
            )
            self.trainset.is_train = False # to avoid data augmentation
            self.testset = DexYCBDataset(
                self.cfg.data_dir,
                is_train=False,
                aug=self.cfg,
                cfg=self.cfg,
            )

        elif self.cfg.dataset_name == 'ho3d':
            self.trainset = HO3DDataset_Train(
                self.cfg.data_dir,
                is_train=True,
                aug=self.cfg,
                cfg=self.cfg,
            )
            self.trainset.is_train = False
            self.testset = HO3DDataset_Test(
                self.cfg.data_dir,
                aug=self.cfg,
                cfg=self.cfg,
            )
        else:
            raise NotImplementedError
        
        self.training_dataloader = DataLoader(
                self.trainset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        
        self.testing_dataloader = DataLoader(
            self.testset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.eval_num_workers,
            pin_memory=False,
            drop_last=False,
        )


    def prepare_accelerator(self):
        self.model, self.optimizer1, self.optimizer2, self.training_dataloader, self.testing_dataloader = self.accel.prepare(
                self.model, self.optimizer1, self.optimizer2, self.training_dataloader, self.testing_dataloader
            )
        
    def get_obj_mesh(self,):
        self.mesh = []
        for k, v in YCB_MESHES.items():
            self.mesh.append(v['verts_sampled'])
        self.mesh = np.stack(self.mesh, axis=0)
        self.mesh = torch.from_numpy(self.mesh).float().to(self.device)

    def optimize_batch(self, ):
        pbar = tqdm(self.training_dataloader, disable=not self.accel.is_local_main_process)
        for batch_idx, batch in enumerate(pbar):
            self.init_param()
            with self.accel.autocast():
                to_device(batch, self.device)

                force_contact = batch['force_contact']
                contact_mask = force_contact > 0.1
                vert3d = batch['gt_hand_vert_flip']
                gravity = batch['gravity']
                CoM = batch['obj_CoM']
                obj_id = batch['obj_id']
                obj_name = batch['obj_name']
                obj_mesh = self.mesh[obj_id].clone()
                obj_r = rotation_6d_to_matrix(batch['gt_obj'][:, :6])
                obj_t = batch['gt_obj'][:, 6:]
                obj_mesh = torch.einsum('bij,bkj->bik', obj_mesh, obj_r) + obj_t[:, None]
                is_grasped = batch['is_grasped']
                rgb_path = batch['rgb_path']
                

                idx_tensor = torch.arange(batch['is_right'].size(0), device=self.device)
                idx_flip = idx_tensor[~batch['is_right']]
                gravity[idx_flip, ..., 0] *= -1
                CoM[idx_flip, ..., 0] *= -1
                obj_mesh[idx_flip, ..., 0] *= -1

                bs = len(rgb_path)


                for i in range(3000):
                    scale = self.scale[:bs].clone()
                    weight = self.weight[:bs].clone()
                    scale = scale * contact_mask
                    if self.accel.num_processes > 1:
                        force_local = self.model.module.get_local_force(scale, weight)
                        force_point, force_global = self.model.module.from_local_to_global(force_local, vert3d)
                    else:
                        force_local = self.model.get_local_force(scale, weight)
                        force_point, force_global = self.model.from_local_to_global(force_local, vert3d)

                    resultant_force = force_global.sum(1, keepdim=True) + gravity
                    resultant_force = resultant_force.squeeze(1)
                    force_loss = torch.norm(resultant_force, dim=-1).mean()
                    sum_weight = force_loss.detach()

                    resultant_force = force_global.sum(1, keepdim=True)
                    cos_proj = torch.einsum('...i,...i->...', resultant_force, -1 * gravity)
                    gravity_loss = F.mse_loss(cos_proj, torch.ones_like(cos_proj))

                    arm = force_point - CoM
                    moment = torch.cross(arm, force_global, dim=-1)
                    moment = moment.sum(1)
                    moment_loss = torch.norm(moment, dim=-1).mean() * 30
                    moment_loss = moment_loss / (100*sum_weight**2 + 1e-8)

                    scale_norm = scale / (scale.norm(dim=-1, keepdim=True).detach() + 1e-8).detach()
                    force_contact_norm = force_contact / (force_contact.norm(dim=-1, keepdim=True).detach() + 1e-8)
                    dist = torch.log(torch.abs(force_contact_norm / (scale_norm + 1e-8))+1e-8) * contact_mask
                    dist_loss = (dist**2).mean() * 0.1
                    dist_loss = dist_loss / (1000*sum_weight**2 + 1e-8)

                    if i < 300:
                        loss = gravity_loss
                        optimizer = self.optimizer1
                    else:
                        loss = force_loss + moment_loss + dist_loss
                        optimizer = self.optimizer2

                    optimizer.zero_grad()
                    self.accel.backward(loss)
                    optimizer.step()

                    info = f"[{i}/{3000}] Loss: {loss.item():.2e}"
                    info += f" | force: {force_loss.item():.2e} gravity: {gravity_loss.item():.2e} "
                    info += f"moment: {moment_loss.item():.2e} dist: {dist_loss.item():.2e}"
                    pbar.set_description(info)

                idx_tensor = torch.arange(batch['is_right'].size(0), device=self.device)
                idx_nograsp = idx_tensor[~is_grasped]
                force_global[idx_nograsp] = 0
                force_local[idx_nograsp] = 0

                self.save_viz(
                    batch_idx=batch_idx,
                    vert=vert3d,
                    force_point=force_point,
                    force_global=force_global,
                    gravity=gravity,
                    CoM=CoM,
                    obj_mesh=obj_mesh,)
                
                self.save_force(
                    force_local=force_local,
                    force_global=force_global,
                    rgb_path=rgb_path,
                )

    def save_viz(self, **kwargs):
        i = self.cfg.eval_batch_size * kwargs['batch_idx']
        j = 120
        k = i*(self.accel.num_processes) + self.accel.process_index + j

        joint = VERT2JOINT(kwargs['vert'][j])
        skeleton = torch.stack([joint[SKELETON[:, 0]], joint[SKELETON[:, 1]]], dim=1)
        gravity = torch.cat([kwargs['CoM'][j], kwargs['CoM'][j] + kwargs['gravity'][j]*0.3], dim=0)[None]
        force = torch.stack([kwargs['force_point'][j], kwargs['force_point'][j]+kwargs['force_global'][j]*0.3, ], axis=1)
        obj_mesh = kwargs['obj_mesh'][j]
        vert = kwargs['vert'][j]

        save_dt = {
            'vert_#00FF00': vert,
            'skeleton_#00FF00': skeleton,
            'force_#FF0000': force,
            'gravity_#0000FF': gravity,
            'obj_mesh_#000000': obj_mesh,
        }
        save_dt = to_numpy(save_dt)

        save_path = os.path.join(self.save_dir, f"viz/{k}_optimized_force.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f: pickle.dump(save_dt, f)

    def save_force(self, **kwargs):
        kwargs = to_numpy(kwargs)

        for i in range(len(kwargs['rgb_path'])):
            save_dt = {
                'force_local': kwargs['force_local'][i],
                'force_global': kwargs['force_global'][i]
            }
            if self.cfg.dataset_name == 'dexycb':
                save_path = kwargs['rgb_path'][i].replace("DexYCB/", "DexYCB/cache/hand_force/").replace('.jpg', '.pkl').replace('color_', 'hand_force_')
            elif self.cfg.dataset_name == 'ho3d':
                save_path = kwargs['rgb_path'][i].replace("HO3D_v2/", "HO3D_v2/cache/hand_force/").replace('.png', '.pkl').replace('rgb/', 'hand_force/')
            else:
                raise NotImplementedError
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f: pickle.dump(save_dt, f)

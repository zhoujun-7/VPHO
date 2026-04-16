import os
import tqdm
import time
import torch
import pickle
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.engine.base_trainer import BaseTrainer
from lib.model._deprecated_net2 import DiffHand
from lib.model._deprecated_net3 import SimpleDiffusion
from lib.configs.args import Config
from lib.dataset._deprecated_dexycb import DexYCBDataset
from lib.utils.misc_fn import to_device, to_numpy
from lib.utils.hand_fn import get_mean_joint_error, get_pa_mean_joint_error

class Trainer(BaseTrainer):
    def __init__(self, cfg:Config):
        super(Trainer, self).__init__(cfg)


    def get_model(self):
        if self.cfg.model == 'diffusion_hand':
            self.model = DiffHand()
        elif self.cfg.model == 'simple_diffusion':
            self.model = SimpleDiffusion()
        else:
            raise ValueError(f"Invalid model name: {self.cfg.model}")

        if self.cfg.pretrain:
            state_dict = torch.load(self.cfg.pretrain)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)


    def get_optimizer(self):
        params = self.model.parameters()
        self.optimizer = optim.AdamW(params, betas=(0.9, 0.999), eps=1e-8, lr=self.cfg.base_learning_rate)     

    def get_scheduler(self):
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.98)

    def get_dataloader(self):
        if self.cfg.dataset_name == 'dexycb':
            self.trainset = DexYCBDataset(
                self.cfg.data_dir,
                is_train=True,
                aug=self.cfg,
                cfg=self.cfg,
            )
            self.testset = DexYCBDataset(
                self.cfg.data_dir,
                is_train=False,
                aug=self.cfg,
                cfg=self.cfg,
            )
        elif self.cfg.dataset_name == 'ho3d':
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid dataset name: {self.cfg.dataset_name}")

        self.training_dataloader = DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.testing_dataloader = DataLoader(
            self.testset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def run(self):
        self.load_checkpoint()
        best_performance = 0
        for epoch in range(self.cfg.max_epochs):
            self.train_one_epoch(epoch)
            self.save_model()
            # performance = self.test()
            self.save_checkpoint(epoch)

            # if performance > best_performance:
            #     best_performance = performance
            #     self.save_model()

    def train_one_epoch(self, epoch):
        self.model.train()
        pbar = tqdm.tqdm(self.training_dataloader)
        for i, batch in enumerate(pbar):
            to_device(batch, self.device)
            with self.accel.accumulate(self.model):
                loss = self.model(batch, is_train=True)

                final_loss = loss['loss_diff_obj']
                self.accel.backward(final_loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                info = f"[{epoch}/{self.cfg.max_epochs}][{i}/{len(self.training_dataloader)}] Loss: {final_loss.item()}"
                pbar.set_description(info)

                if i % self.cfg.print_freq == 0:
                    self.logger.info(info)
        self.scheduler.step()

    # def test_obj(self):
    #     self.model.eval()
    #     pbar = tqdm.tqdm(self.testing_dataloader)
    #     all_mje = []
    #     all_pa_mje = []

    #     for i, batch in enumerate(pbar):
    #         with torch.no_grad():
    #             to_device(batch, self.device)
    #             res = self.model(batch, is_train=False)
        
    #         res = to_numpy(res)
    #         batch = to_numpy(batch)
    #         mje = get_mean_joint_error(res['joints'], batch['gt_jt3d']) * 1000
    #         pa_mje = get_pa_mean_joint_error(res['joints'], batch['gt_jt3d']) * 1000

    #         info = f"[{i}/{len(self.training_dataloader)}] MJE: {mje.mean()} PAMJE: {pa_mje.mean()}"
    #         pbar.set_description(info)
    #         if i % 10 == 0:
    #             self.logger.info(info)
            
    #         all_mje.append(mje)
    #         all_pa_mje.append(pa_mje)

    #         if i == 100:
    #             break

    #     all_mje = np.array(all_mje).mean()
    #     all_pa_mje = np.array(all_pa_mje).mean()
    #     return all_mje

    @ torch.no_grad()
    def infer_energy(self):
        from pytorch3d.transforms import matrix_to_rotation_6d
        pred_obj_pose_path = "/root/Workspace/HOI/FeHO/HOI/HFL-Net2/host_folder/dex-ycb_from_zero_after25epoch7/all_diff_result.pkl"
        with open(pred_obj_pose_path, 'rb') as f:
            pred_obj_pose = pickle.load(f)

        self.model.eval()
        pbar = tqdm.tqdm(self.testing_dataloader)
        
        B = self.cfg.batch_size
        save_dt = {}
        for i, batch in enumerate(pbar):
            bs = batch['rgb'].shape[0]
            pd_obj_pose = []
            for j in range(bs):
                idx = i * B + j
                pd_obj_pose.append(pred_obj_pose[idx])
            pd_obj_pose = np.stack(pd_obj_pose, axis=0)
            pd_obj_pose = torch.from_numpy(pd_obj_pose).float()
            
            root_joint = batch['root_joint']
            obj_rot6d = matrix_to_rotation_6d(pd_obj_pose[..., :3, :3])
            pd_obj_pose = torch.cat([obj_rot6d, pd_obj_pose[..., :3, 3] - root_joint[:, None]], dim=-1)
            batch['sampled_pose'] = pd_obj_pose
            to_device(batch, self.device)
            score = self.model(batch, mode='energy')
            if save_score := True:
                for j in range(bs):
                    save_dt[f"{i*B + j}"] = score[j].cpu().numpy()
        save_path = "/root/Workspace/HOI/FeHO/HOI/HFL-Net2/host_folder/dex-ycb_from_zero_after25epoch7/all_diff_result_energy.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(save_dt, f)
        print(f"Save energy to {save_path}")

    def test(self):
        self.model.eval()
        pbar = tqdm.tqdm(self.testing_dataloader)
        all_mje = []
        all_pa_mje = []

        for i, batch in enumerate(pbar):
            with torch.no_grad():
                to_device(batch, self.device)
                res = self.model(batch, is_train=False)

                save_dt = {
                    'verts': res['verts'][0].detach().cpu().numpy(),
                    'gt_hand_vert': batch['gt_hand_vert'][0].detach().cpu().numpy(),
                }
                np.savez(f"tmp/{i}.npz", **save_dt)
        
            res = to_numpy(res)
            batch = to_numpy(batch)
            mje = get_mean_joint_error(res['joints'], batch['gt_jt3d']) * 1000
            pa_mje = get_pa_mean_joint_error(res['joints'], batch['gt_jt3d']) * 1000

            info = f"[{i}/{len(self.training_dataloader)}] MJE: {mje.mean()} PAMJE: {pa_mje.mean()}"
            pbar.set_description(info)
            if i % 10 == 0:
                self.logger.info(info)
            
            all_mje.append(mje)
            all_pa_mje.append(pa_mje)

            if i == 100:
                break

        all_mje = np.array(all_mje).mean()
        all_pa_mje = np.array(all_pa_mje).mean()
        return all_mje

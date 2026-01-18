import os
import cv2
import json
from tqdm.auto import tqdm
import torch
import pickle
import pprint
import numpy as np
import pandas as pd
from copy import copy, deepcopy
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from lib.engine.base_trainer import BaseTrainer
from lib.model.VPHO import vpho_net
from lib.model.physics import from_local_to_global
from lib.configs.args import Config
from lib.dataset.dexycb6 import DexYCBDataset_Force
from lib.dataset.base import inverse_normalize_rgb
from lib.utils.misc_fn import to_device, to_numpy, to_tensor
from lib.utils.transform_fn import obj_9D_to_mat, obj_mat_to_9D, OPENGL_TO_OPENCV
from lib.engine.test import TesterObject, TesterHand
from lib.dataset.base import YCB_MESHES, mano_layer_r
from lib.utils.viz_fn import get_random_color, make_heatmaps, depth_to_rgb


class Trainer(BaseTrainer):
    def __init__(self, cfg:Config):
        super(Trainer, self).__init__(cfg)
        self.tester_obj = TesterObject()
        self.tester_hand = TesterHand()

    def get_model(self):
        if self.cfg.model == 'vpho_net':
            self.model = vpho_net()
        else:
            raise ValueError(f"Invalid model name: {self.cfg.model}")

        if self.cfg.pretrain:
            state_dict = torch.load(self.cfg.pretrain)
            rm_key_ls = self.cfg.remove_pretrained_keys
            for k in list(state_dict.keys()):
                if any([k.startswith(rm_key) for rm_key in rm_key_ls]):
                    del state_dict[k]
                    self.logger.warning(f"Remove pretrained state dict: {k}", main_process_only=True)
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)

    def get_optimizer(self):
        params = self.model.parameters()
        if self.cfg.optimizer == 'adamw':
            self.optimizer = optim.AdamW(params, betas=(0.9, 0.999), eps=1e-8, lr=self.cfg.base_learning_rate)     
        elif self.cfg.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=self.cfg.base_learning_rate, weight_decay=0.0005)

    def get_scheduler(self):
        if self.cfg.scheduler == 'exp':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.gamma**(1/self.accel.num_processes)) #! Accelerate will scale the scheduler step size by the number of processes
        elif self.cfg.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer, 
                max_lr=self.cfg.base_learning_rate, 
                epochs=self.cfg.max_epochs,
                steps_per_epoch=len(self.training_dataloader),
                pct_start=0.1, 
                anneal_strategy='cos',
                last_epoch=(self.start_epoch) * len(self.training_dataloader) if self.start_epoch > 1 else -1
            )
        elif self.cfg.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.lr_step, gamma=self.cfg.gamma**(1/self.accel.num_processes))
        else:
            raise ValueError(f"Invalid scheduler name: {self.cfg.scheduler}")

    def get_dataloader(self):
        if self.cfg.dataset_name == 'dexycb':
            self.trainset = DexYCBDataset_Force(
                self.cfg.data_dir,
                is_train=True,
                aug=self.cfg,
                cfg=self.cfg,
            )
            self.testset = DexYCBDataset_Force(
                self.cfg.data_dir,
                is_train=False,
                aug=self.cfg,
                cfg=self.cfg,
            )
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
                batch_size=self.cfg.eval_batch_size,
                shuffle=False,
                num_workers=self.cfg.eval_num_workers,
                pin_memory=False,
                drop_last=False,
            )
            sub_index = np.arange(0, len(self.testset), 10)
            sub_testset = Subset(self.testset, sub_index)
            self.trainset.is_train = False
            self.sub_testing_dataloader = DataLoader(
                sub_testset,
                batch_size=self.cfg.eval_batch_size,
                shuffle=True,
                num_workers=self.cfg.eval_num_workers,
                pin_memory=False,
                drop_last=False,
                generator=torch.Generator().manual_seed(128),
            )
        
        else:
            raise ValueError(f"Invalid dataset name: {self.cfg.dataset_name}")

    def prepare_accelerator(self):
        self.model, self.optimizer, self.training_dataloader, self.testing_dataloader, self.sub_testing_dataloader, self.scheduler = self.accel.prepare(
                self.model, self.optimizer, self.training_dataloader, self.testing_dataloader, self.sub_testing_dataloader, self.scheduler
            )

    def run(self):
        if self.cfg.checkpoint:
            self.load_checkpoint()
        if self.cfg.start_with_eval:
            if self.cfg.dataset_name == 'dexycb':
                self.evaluate(self.sub_testing_dataloader)
            elif self.cfg.dataset_name == 'ho3d':
                self.infer()
            else:
                raise NotImplementedError
            
        for epoch in range(self.start_epoch, self.cfg.max_epochs):
            info = f"Epoch: {epoch}/{self.cfg.max_epochs} | Learning Rate: {self.optimizer.param_groups[0]['lr']:.3e}"
            self.logger.info(info, main_process_only=True)
            self.accel.wait_for_everyone()
            
            self.train_one_epoch(epoch)
            self.save_checkpoint(epoch+1)

            self.logger.info(f"Evaluation on subtest dataset: {len(self.sub_testing_dataloader.dataset)}", main_process_only=True)
            if self.cfg.dataset_name != 'ho3d':
                self.evaluate(self.sub_testing_dataloader) 
            elif self.cfg.dataset_name == 'ho3d':
                if (epoch+1) % self.cfg.full_evaluation_freq == 0:
                    if self.cfg.use_mix_trainset:
                        self.evaluate(self.sub_testing_dataloader) 
                    self.infer(epoch+1)
            self.save_model()

    def eval(self,):
        if self.cfg.checkpoint:
            self.load_checkpoint()
        
        if self.cfg.dataset_name == 'dexycb':
            if self.cfg.eval_full:
                self.evaluate(self.testing_dataloader)
            else:
                self.evaluate(self.sub_testing_dataloader)
        elif self.cfg.dataset_name == 'ho3d':
            self.infer()
        else:
            raise NotImplementedError

    def train_one_epoch(self, epoch):
        self.accel.wait_for_everyone()
        self.model.train()
        pbar = tqdm(self.training_dataloader, disable=not self.accel.is_local_main_process)
        for i, batch in enumerate(pbar):
            with self.accel.autocast():
                to_device(batch, self.device)
                with self.accel.accumulate(self.model):
                    loss_dt, pd_dt = self.model(batch, mode='train')

                    final_loss = loss_dt['total_loss']
                    self.accel.backward(final_loss)

                    if self.cfg.gradient_clip > 0 and self.accel.sync_gradients:
                        self.accel.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    info = f"[{i:04d}/{len(self.training_dataloader)}]"
                    info += "| Loss"
                    # TODO: reduce loss_dt
                    for c, (k, v) in enumerate(loss_dt.items()):
                        info += f" {k.replace('_loss', '').replace('hand', 'H').replace('obj', 'O').replace('reg', 'R')}:{v.item():.2e}"

                if i % self.cfg.print_freq == 0:
                    if self.accel.is_main_process: print('\r', end='')
                    self.logger.info(info, main_process_only=True)
                pbar.set_description(info)
    
            if self.scheduler == 'cosine': self.scheduler.step()
        if self.scheduler != 'cosine': self.scheduler.step()


    @torch.no_grad()
    def evaluate(self, testing_dataloader):
        self.accel.wait_for_everyone()
        self.model.eval()
        pbar = tqdm(testing_dataloader, disable=not self.accel.is_local_main_process)
        
        collector_hand = []
        collector_obj = []
        collector_force = []
        collector_physics = []
        collector_res = []
        for i, batch in enumerate(pbar):

            # if i == 10:
            #     break
            to_device(batch, self.device)
            self.accel.wait_for_everyone() #! Even if the model is in eval mode, it is necessary to synchronize the model, otherwise the other processes will be very slow.
            
            if i == 0:
                compute_flops(self.model, batch)
            
            res_dt = self.model(batch, mode='predict')
            res_dt = self.postprocess(res_dt, batch['root_joint'], batch['is_right'])
            batch = self.postprocess(batch, batch['root_joint'])
            res_dt = to_numpy(res_dt)
            batch = to_numpy(batch)
            

            hand_data = res_dt['agg_hand_mano']
            obj_data = res_dt['agg_obj_rt']
            data_path = batch['rgb_path']
            
            res_dt = to_numpy(res_dt)
            batch = to_numpy(batch)

            eval_hand_dt = {
                'is_right': batch['is_right'],
                'gt_joint': batch['gt_joint'],
                'gt_vert': batch['gt_hand_vert'],
                'pd_vert_reg': res_dt['reg_hand_vert'],
                'pd_joint_reg': res_dt['reg_hand_joint'],
                'pd_vert_diff_final': res_dt['diff_final_hand_vert'],
                'pd_joint_diff_final': res_dt['diff_final_hand_joint'],
                'pd_vert_diff_agg': res_dt['agg_hand_vert'],
                'pd_joint_diff_agg': res_dt['agg_hand_joint'],
            }
            collector_hand.append(self.test_diff_hand(eval_hand_dt, is_eval_best=False))
            eval_obj_dt = {
                'pd_rt': res_dt['diff_final_obj_rt'],
                'gt_rt': batch['gt_obj_rt'],
                'cam_intr': batch['cam_intr'],
                'obj_name': batch['obj_name'],
                'pd_rt_agg': res_dt['agg_obj_rt'],
            }
            if 'reg_obj_rt' in res_dt:
                eval_obj_dt['pd_rt_reg'] = res_dt['reg_obj_rt']
            collector_obj.append(self.test_diff_object(eval_obj_dt, is_eval_best=False))

            pd_res_dt = {
                'index': batch['index'],
                'path': batch['rgb_path'],
                'pd_obj_rt': res_dt['agg_obj_rt'],
                'pd_hand_vert': res_dt['agg_hand_vert'].astype(np.float16),
                'pd_hand_joint': res_dt['agg_hand_joint'],
                'gt_obj_rt': batch['gt_obj_rt'],
                'obj_name': batch['obj_name'],
            }
            collector_res.append(pd_res_dt)


            if self.cfg.viz_freq > 0:
                if testing_dataloader == self.sub_testing_dataloader:
                    viz_i = i * 10
                elif testing_dataloader == self.testing_dataloader:
                    viz_i = i
                else:
                    raise ValueError("Invalid testing dataloader")
                if viz_i % self.cfg.viz_freq == 0:
                    self.save_viz_hand(
                        rgb_path=batch['rgb_path'],
                        obj_name=batch['obj_name'],
                        gt_vert=batch['gt_hand_vert'],
                        pd_vert_reg=res_dt['reg_hand_vert'],
                        pd_vert_diff_final=res_dt['diff_final_hand_vert'],
                        pd_vert_diff_inprocess=res_dt['diff_inprocess_hand_vert'],
                        pd_vert_diff_agg=res_dt['agg_hand_vert'],
                        gt_obj_rt=batch['gt_obj_rt'],
                        batch_idx=viz_i)
                    self.save_viz_obj(
                        rgb_path=batch['rgb_path'],
                        pd_rt=res_dt['diff_final_obj_rt'],
                        gt_rt=batch['gt_obj_rt'],
                        pd_rt_agg=res_dt['agg_obj_rt'],
                        pd_inprocess_rt=res_dt['diff_inprocess_obj_rt'],
                        pd_rt_reg=res_dt['reg_obj_rt'] if 'reg_obj_rt' in res_dt else None,
                        obj_name=batch['obj_name'],
                        gt_hand=batch['gt_hand_vert'],
                        batch_idx=viz_i)
                    self.save_viz_heatmap(
                        rgb=batch['rgb'],
                        pd_hand_heatmap=res_dt['hand_heatmap'],
                        gt_hand_heatmap=batch['hm_hand'],
                        pd_obj_heatmap=res_dt['obj_heatmap'],
                        gt_obj_heatmap=batch['hm_obj'],
                        gt_hand_bbox=batch['bbox_hand'],
                        gt_obj_bbox=batch['bbox_obj'],
                        batch_idx=viz_i)
                    if 'obj_segm' in res_dt:
                        self.save_viz_segmentation(
                            rgb=batch['rgb'],
                            pd_obj_segm=res_dt['obj_segm'],
                            gt_obj_segm=batch['segm_obj'],
                            gt_obj_bbox=batch['bbox_obj'],
                            batch_idx=viz_i)
                    if 'force_local' in res_dt:
                        self.save_viz_force(
                            rgb_path=batch['rgb_path'],
                            pd_force_local=res_dt['force_local'],
                            gt_force_local=batch['force_local'],
                            pd_hand_vert=res_dt['agg_hand_vert'],
                            gt_hand_vert=batch['gt_hand_vert_flip'],
                            obj_name=batch['obj_name'],
                            pd_obj_rt=res_dt['agg_obj_rt'],
                            gt_obj_rt=batch['gt_obj_rt'],
                            gravity=batch['gravity'],
                            root_joint=batch['root_joint'],
                            is_right=batch['is_right'],
                            batch_idx=viz_i)
            pbar.set_description('Evaluation')

        # gather all gpus
        collector_hand = self.accel.gather_for_metrics(collector_hand, use_gather_object=True)
        collector_obj = self.accel.gather_for_metrics(collector_obj, use_gather_object=True)
        collector_res = self.accel.gather_for_metrics(collector_res, use_gather_object=True)
        if self.accel.is_main_process:
            mean_collector_hand, collector_hand = self.__collect_dict(collector_hand)
            num_sample = collector_hand['regression']['MJE']['both'].shape[0]
            self.logger.info(f"Hand Evaluation: {num_sample} samples", main_process_only=True)
            for k, v in mean_collector_hand.items():
                df = pd.DataFrame(v)
                df = df.map(lambda x: f"{1000*x:.2f}")
                df = df.to_string()
                info = f"{k}: \n{df}"
                self.logger.info(info, main_process_only=True)

            mean_collector_obj, collector_obj = self.__collect_dict(collector_obj)
            mean_pose_df = obj_dt_to_dataframe(mean_collector_obj["mean_candidate_pose"])
            mean_pose_df = mean_pose_df.to_string()
            info = f"Object Evaluation: {num_sample} samples \n"
            info += f"Mean Pose: \n{mean_pose_df}"

            if mean_collector_obj['regression']:
                reg_pose_df = obj_dt_to_dataframe(mean_collector_obj['regression'])
                reg_pose_df = reg_pose_df.to_string()
                info += f"\nRegression Pose: \n{reg_pose_df}"
            self.logger.info(info, main_process_only=True)

    @torch.no_grad()
    def infer(self, epoch=""):
        self.accel.wait_for_everyone()
        self.model.eval()
        pbar = tqdm(self.infering_dataloader, disable=not self.accel.is_local_main_process)

        collector_hand = []
        collector_obj = []
        collector_res = []
        for i, batch in enumerate(pbar):
            to_device(batch, self.device)
            self.accel.wait_for_everyone() 

            res_dt = self.model(batch, mode='predict')
            res_dt = self.postprocess(res_dt, batch['root_joint'], batch['is_right'])
            batch = self.postprocess(batch, batch['root_joint'])
            res_dt = to_numpy(res_dt)
            batch = to_numpy(batch)
            
            phy_data_dt = {
                'index': batch['index'],
                'path': batch['rgb_path'],
                'pd_obj_rt': res_dt['agg_obj_rt'],
                # 'pd_obj_rt': res_dt['reg_obj_rt'],
                'pd_hand_vert': res_dt['agg_hand_vert'].astype(np.float16),
                'pd_hand_joint': res_dt['agg_hand_joint'],
            }

            res_dt['reg_hand_joint'] = res_dt['reg_hand_joint'] @ OPENGL_TO_OPENCV
            res_dt['reg_hand_vert'] = res_dt['reg_hand_vert'] @ OPENGL_TO_OPENCV
            res_dt['agg_hand_joint'] = res_dt['agg_hand_joint'] @ OPENGL_TO_OPENCV
            res_dt['agg_hand_vert'] = res_dt['agg_hand_vert'] @ OPENGL_TO_OPENCV

            collector_res.append(phy_data_dt)

            for ind in range(len(batch['index'])):
                eval_hand_dt = {
                    batch['index'][ind].item(): {
                        'joint_reg': res_dt['reg_hand_joint'][ind],
                        'vert_reg': res_dt['reg_hand_vert'][ind],
                        'joint_diff': res_dt['agg_hand_joint'][ind],
                        'vert_diff': res_dt['agg_hand_vert'][ind],
                    }
                }
                collector_hand.append(eval_hand_dt)

            eval_obj_dt = {
                'pd_rt': res_dt['diff_final_obj_rt'],
                'gt_rt': batch['gt_obj_rt'],
                'cam_intr': batch['cam_intr'],
                'obj_name': batch['obj_name'],
                'pd_rt_agg': res_dt['agg_obj_rt'],
            }
            collector_obj.append(self.test_diff_object(eval_obj_dt, is_eval_best=False))

            pbar.set_description('Inference')

        collector_hand = self.accel.gather_for_metrics(collector_hand, use_gather_object=True)
        collector_obj = self.accel.gather_for_metrics(collector_obj, use_gather_object=True)
        collector_res = self.accel.gather_for_metrics(collector_res, use_gather_object=True)
        if self.accel.is_main_process:
            collector_hand = {k: v for dt in collector_hand for k, v in dt.items()}
            joint_reg, vert_reg, joint_diff, vert_diff = [], [], [], []
            for i in range(len(collector_hand)):
                joint_reg.append(collector_hand[i]['joint_reg'])
                vert_reg.append(collector_hand[i]['vert_reg'])
                joint_diff.append(collector_hand[i]['joint_diff'])
                vert_diff.append(collector_hand[i]['vert_diff'])
            hand_reg_path = os.path.join(self.save_dir, 'submit', f'{epoch}hand_reg.json')
            hand_diff_path = os.path.join(self.save_dir, 'submit', f'{epoch}hand_diff.json')
            os.makedirs(os.path.dirname(hand_reg_path), exist_ok=True)
            dump(hand_reg_path, joint_reg, vert_reg)
            dump(hand_diff_path, joint_diff, vert_diff)
            os.system(f"zip -j {hand_reg_path.replace('.json', '.zip')} {hand_reg_path}")
            os.system(f"zip -j {hand_diff_path.replace('.json', '.zip')} {hand_diff_path}")
            os.system(f"rm {hand_reg_path}")
            os.system(f"rm {hand_diff_path}")

            mean_collector_obj, collector_obj = self.__collect_dict(collector_obj)
            mean_pose_df = obj_dt_to_dataframe(mean_collector_obj["mean_candidate_pose"])
            info = f"Object Evaluation: \n"
            info += f"Mean Pose: \n{mean_pose_df}"
            self.logger.info(info, main_process_only=True)

            with open(f"{self.save_dir}/my-prediction_align-{self.cfg.clean_data_mode}.pkl", "wb") as f:
                pickle.dump(collector_res, f)
            
    def test_object(self, eval_dt):
        test_dt = self.tester_obj(eval_dt)

        info = ""
        for k, v in test_dt.items():
            info += f"{k}: {pprint.pformat(v, indent=4)} \n"
        self.logger.info(info, main_process_only=True)

    def test_diff_hand(self, eval_dt, is_eval_best=False):
        # eval regression
        eval_reg_dt = {
            'is_right': eval_dt['is_right'],
            'gt_joint': eval_dt['gt_joint'],
            'pd_joint': eval_dt['pd_joint_reg'],
            'gt_vert': eval_dt['gt_vert'],
            'pd_vert': eval_dt['pd_vert_reg'],
        }
        out_reg_dt = self.tester_hand(eval_reg_dt)

        # evaluate one candidate
        eval_1_dt = copy(eval_dt)
        eval_1_dt['pd_joint'] = eval_1_dt['pd_joint_diff_final'][:, 0]
        eval_1_dt['pd_vert'] = eval_1_dt['pd_vert_diff_final'][:, 0]
        out_1_dt = self.tester_hand(eval_1_dt)

        eval_agg_mano_dt = copy(eval_reg_dt)
        eval_agg_mano_dt['pd_joint'] = eval_dt['pd_joint_diff_agg']
        eval_agg_mano_dt['pd_vert'] = eval_dt['pd_vert_diff_agg']
        out_agg_mano_dt = self.tester_hand(eval_agg_mano_dt)
        
        # evaluate the best candidate
        if is_eval_best:
            eval_best_dt = copy(eval_dt)
            eval_best_dt['pd_joint_diff_final'] = eval_dt['pd_joint_diff_final']
            eval_best_dt['pd_vert_diff_final'] = eval_dt['pd_vert_diff_final']
            out_best_dt = self.tester_hand(eval_best_dt)
        else:
            out_best_dt = {}
        return {
            'regression': out_reg_dt,
            'one_candidate': out_1_dt,
            # 'mean_candidate_joint': out_mean_joint_dt,
            'agg_candidate': out_agg_mano_dt,
            # 'best_candidate': out_best_dt,
        }

    def test_diff_object(self, eval_dt, is_eval_best=False):
        # evaluate one candidate
        eval_1_dt = deepcopy(eval_dt)
        eval_1_dt['pd_rt'] = eval_1_dt['pd_rt'][:, 0]
        out_1_dt = self.tester_obj(eval_1_dt)

        # evaluate the mean pose of all candidates
        eval_mean_pose_dt = deepcopy(eval_dt)
        eval_mean_pose_dt['pd_rt'] = eval_dt['pd_rt_agg']
        out_mean_pose_dt = self.tester_obj(eval_mean_pose_dt)

        # evaluate the best candidate
        if is_eval_best:
            out_best_dt = self.tester_obj(eval_dt)
        else:
            out_best_dt = {}

        # evaluate regression
        if 'pd_rt_reg' in eval_dt:
            eval_reg = deepcopy(eval_dt)
            eval_reg['pd_rt'] = eval_dt['pd_rt_reg']
            out_reg_dt = self.tester_obj(eval_reg)
        else:
            out_reg_dt = {}

        return {
            'one_candidate': out_1_dt,
            # 'mean_candidate_vert': out_mean_vert_dt,
            'mean_candidate_pose': out_mean_pose_dt,
            'best_candidate_pose': out_best_dt,
            'regression': out_reg_dt,
        }

    def __pprint_dict(self, dt):
        info = ""
        for k, v in dt.items():
            info += f"{k}: {pprint.pformat(v, indent=4)} \n"
        self.logger.info(info, main_process_only=True)

    def __collect_dict(self, ls):
        def iter_collect_dt(dt1, dt2):
            for k, v in dt2.items():
                if isinstance(v, dict):
                    if k not in dt1:
                        dt1[k] = {}
                    iter_collect_dt(dt1[k], dt2[k])
                else:
                    if k not in dt1:
                        dt1[k] = v
                    else:
                        dt1[k] = np.concatenate([dt1[k], dt2[k]], axis=0)
        def iter_average_dt(dt):
            for k, v in dt.items():
                if isinstance(v, dict):
                    iter_average_dt(v)
                elif isinstance(v, np.ndarray):
                    if v.shape[0] > 0:
                        dt[k] = v.mean()
                    else:
                        dt[k] = -0.001
        
        collector = {}
        for dt in ls:
            iter_collect_dt(collector, dt)
        mean_collect_dt = deepcopy(collector)
        iter_average_dt(mean_collect_dt)
        return mean_collect_dt, collector

    def __collect_force_dt(self, ls):
        collector = {}
        for dt in ls:
            for k, v in dt.items():
                if k not in collector:
                    collector[k] = []
                collector[k].append(v)
        for k, v in collector.items():
            collector[k] = np.concatenate(v, axis=0)

        mean_collect_dt = {}
        for k, v in collector.items():
            if v.shape[0] > 0:
                mean_collect_dt[k] = v.mean()
            else:
                mean_collect_dt[k] = -1
        return mean_collect_dt, collector

    def postprocess(self, data, root_joint, is_right=None):
        for k in list(data.keys()):
            if k in ['diff_inprocess_obj_6d', 'diff_final_obj_6d', 'gt_obj', 'agg_obj_6d', 'reg_obj_6d']:
                new_k = k.replace('_6d', '')
                data[f'{new_k}_rt'] = self.__postprocess_obj_rt(data[k], root_joint)

        for k, v in data.items():
            if k in ['reg_hand_vert', 'reg_hand_joint', 'diff_final_hand_vert', 'diff_final_hand_joint', 'agg_hand_vert', 'agg_hand_joint']:
                data[k] = self.__postprocess_hand_vert(v, root_joint, is_right)
            elif k in ['diff_inprocess_hand_vert', 'diff_inprocess_hand_joint']:
                _is_right = torch.zeros(v.shape[0], dtype=torch.bool, device=v.device) + is_right[0]
                _root_joint = torch.zeros([v.shape[0], root_joint.shape[-1]], device=v.device) + root_joint[0]
                data[k] = self.__postprocess_hand_vert(v, _root_joint, _is_right)
        return data

    def __postprocess_obj_rt(self, rot6d, root_joint):
        rt = obj_9D_to_mat(rot6d)
        rt[..., 3] = torch.einsum("b...i,bi->b...i", torch.ones_like(rt[..., 3]), root_joint) + rt[..., 3]
        return rt
    
    def __postprocess_hand_vert(self, vert, root_joint, is_right):
        flipped_idx = torch.arange(vert.shape[0], device=is_right.device)[~is_right]
        vert[flipped_idx, ..., 0] = -vert[flipped_idx, ..., 0]
        vert = torch.einsum("b...i,bi->b...i", torch.ones_like(vert), root_joint) + vert
        return vert
    
    def save_viz_obj(self, batch_idx, **data):
        i = self.cfg.eval_batch_size * batch_idx
        j = 0
        k = i*(self.accel.num_processes) + self.accel.process_index + j
        obj_vert = YCB_MESHES[data['obj_name'][j]]["verts_sampled"]
        gt_rt = data['gt_rt'][j]
        pd_rt = data['pd_rt'][j]
        pd_inprocess_rt = data['pd_inprocess_rt'][j]
        pd_rt_agg = data['pd_rt_agg'][j]
        pd_rt_reg = data['pd_rt_reg'][j] if data['pd_rt_reg'] is not None else None

        gt_vert = obj_vert @ gt_rt[:3, :3].T + gt_rt[:3, 3]
        pd_vert = np.einsum("ni,...ij->...nj", obj_vert, pd_rt[..., :3, :3].swapaxes(-1, -2)) + pd_rt[..., 3][:, None]
        inprocess_vert = np.einsum("ni,...ij->...nj", obj_vert, pd_inprocess_rt[0, ..., :3, :3].swapaxes(-1, -2)) + pd_inprocess_rt[0, ..., 3][:, None]
        pd_vert_mean = obj_vert @ pd_rt_agg[:3, :3].T + pd_rt_agg[:3, 3]
        pd_vert_reg = obj_vert @ pd_rt_reg[:3, :3].T + pd_rt_reg[:3, 3] if pd_rt_reg is not None else None
        
        obj_name = data['obj_name'][j]
        gt_hand = data['gt_hand'][j]
        rgb_path = data['rgb_path'][j]
        
        multihyperthesis_dt = {'rgb_path': rgb_path, 'obj_name': obj_name, 'gt_hand_#000000': gt_hand, 
                               'obj_gt_vert_#00FF00': gt_vert, 'obj_pd_vert_mean_#FF0000': pd_vert_mean,
                                'obj_pd_vert_reg_#00FFFF': pd_vert_reg}
        for s in range(min(pd_vert.shape[0], 20)):
            color = get_random_color(is_HEX=True, exclude=np.array([0, 255, 0]))
            multihyperthesis_dt[f'obj_diff_vert_{s}_{color}'] = pd_vert[s]

        save_path = os.path.join(self.save_dir, f"viz/{k}_obj_multihyperthesis.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(multihyperthesis_dt, f)

    def save_viz_hand(self, batch_idx, **data):
        i = self.cfg.eval_batch_size * batch_idx
        j = 0
        k = i*(self.accel.num_processes) + self.accel.process_index + j
        rgb_path = data['rgb_path'][j]
        obj_vert = YCB_MESHES[data['obj_name'][j]]["verts_sampled"]
        gt_obj_rt = data['gt_obj_rt'][j]
        gt_obj_vert = obj_vert @ gt_obj_rt[:3, :3].T + gt_obj_rt[:3, 3]

        gt_vert = data['gt_vert'][j]
        pd_vert_reg = data['pd_vert_reg'][j]
        pd_vert_diff = data['pd_vert_diff_final'][j]
        pd_vert_diff_inprocess = data['pd_vert_diff_inprocess']
        pd_vert_diff_agg = data['pd_vert_diff_agg'][j]

        reg_dt = {'rgb_path': rgb_path, 'gt_hand_#000000': gt_vert, 'gt_obj_#00FF00': gt_obj_vert, 
                  'pd_vert_reg_#00FF00': pd_vert_reg, 'pd_vert_diff_agg_#FF0000': pd_vert_diff_agg}
        save_path = os.path.join(self.save_dir, f"viz/{k}_hand_reg_&_diff_mean.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f: pickle.dump(reg_dt, f)

        multi_hyperthesis_dt = {'rgb_path': rgb_path, 'gt_hand_#000000': gt_vert, 'gt_obj_#00FF00': gt_obj_vert, 
                                'pd_vert_reg_#00FF00': pd_vert_reg, 'pd_vert_diff_agg_#FF0000': pd_vert_diff_agg}
        for s in range(min(pd_vert_diff.shape[0], 20)):
            color = get_random_color(is_HEX=True, exclude=np.array([0, 255, 0]))
            multi_hyperthesis_dt[f'hand_diff_vert_{s}_{color}'] = pd_vert_diff[s]
        save_path = os.path.join(self.save_dir, f"viz/{k}_hand_multihyperthesis.pkl")
        with open(save_path, 'wb') as f: pickle.dump(multi_hyperthesis_dt, f)

    def save_viz_heatmap(self, batch_idx, **data):
        i = self.cfg.eval_batch_size * batch_idx
        j = 0
        k = i*(self.accel.num_processes) + self.accel.process_index + j
        rgb = data['rgb'][j, ::-1].transpose(1, 2, 0)
        rgb = inverse_normalize_rgb(rgb)

        gt_hand_bbox = data['gt_hand_bbox'][j].astype(np.int64)
        gt_hand_heatmap = data['gt_hand_heatmap'][j]
        pd_hand_heatmap = data['pd_hand_heatmap'][j]
        gt_hand_heatmap = gt_hand_heatmap.transpose(1, 2, 0)
        gt_hand_heatmap = cv2.resize(gt_hand_heatmap, (pd_hand_heatmap.shape[1], pd_hand_heatmap.shape[2]))
        gt_hand_heatmap = gt_hand_heatmap.transpose(2, 0, 1)
        rgb_hand_crop = rgb[gt_hand_bbox[1]:gt_hand_bbox[3], gt_hand_bbox[0]:gt_hand_bbox[2]]
        rgb_hand_crop = cv2.resize(rgb_hand_crop, (pd_hand_heatmap.shape[-2], pd_hand_heatmap.shape[-1]))
        gt_hand_viz_map = make_heatmaps(rgb_hand_crop, gt_hand_heatmap)
        pd_hand_viz_map = make_heatmaps(rgb_hand_crop, pd_hand_heatmap)
        hand_viz_map = np.concatenate([gt_hand_viz_map, pd_hand_viz_map], axis=0)
        save_path = os.path.join(self.save_dir, f"viz/{k}_gt&pd_hand_heatmap.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, hand_viz_map)

        gt_obj_bbox = data['gt_obj_bbox'][j].astype(np.int64)
        pd_obj_heatmap = data['pd_obj_heatmap'][j]
        gt_obj_heatmap = data['gt_obj_heatmap'][j]
        gt_obj_heatmap = gt_obj_heatmap.transpose(1, 2, 0)
        gt_obj_heatmap = cv2.resize(gt_obj_heatmap, (pd_obj_heatmap.shape[1], pd_obj_heatmap.shape[2]))
        gt_obj_heatmap = gt_obj_heatmap.transpose(2, 0, 1)
        rgb_obj_crop = rgb[gt_obj_bbox[1]:gt_obj_bbox[3], gt_obj_bbox[0]:gt_obj_bbox[2]]
        rgb_obj_crop = cv2.resize(rgb_obj_crop, (pd_obj_heatmap.shape[-2], pd_obj_heatmap.shape[-1]))
        gt_obj_viz_map = make_heatmaps(rgb_obj_crop, gt_obj_heatmap)
        pd_obj_viz_map = make_heatmaps(rgb_obj_crop, pd_obj_heatmap)
        obj_viz_map = np.concatenate([gt_obj_viz_map, pd_obj_viz_map], axis=0)
        save_path = os.path.join(self.save_dir, f"viz/{k}_gt&pd_obj_heatmap.jpg")
        cv2.imwrite(save_path, obj_viz_map)

    def save_viz_segmentation(self, batch_idx, **data):
        i = self.cfg.eval_batch_size * batch_idx
        j = 0
        k = i*(self.accel.num_processes) + self.accel.process_index + j
        rgb = data['rgb'][j, ::-1].transpose(1, 2, 0)
        rgb = inverse_normalize_rgb(rgb)

        gt_obj_bbox = data['gt_obj_bbox'][j].astype(np.int64)
        pd_obj_segm = data['pd_obj_segm'][j, 0]
        gt_obj_segm = data['gt_obj_segm'][j, 0]
        pd_obj_segm = depth_to_rgb(pd_obj_segm)
        gt_obj_segm = depth_to_rgb(gt_obj_segm)

        rgb_obj_crop = rgb[gt_obj_bbox[1]:gt_obj_bbox[3], gt_obj_bbox[0]:gt_obj_bbox[2]]
        rgb_obj_crop = cv2.resize(rgb_obj_crop, (pd_obj_segm.shape[0], pd_obj_segm.shape[1]))
        obj_viz_map = np.concatenate([rgb_obj_crop, gt_obj_segm, pd_obj_segm], axis=1)
        save_path = os.path.join(self.save_dir, f"viz/{k}_gt&pd_obj_segmentation.jpg")
        cv2.imwrite(save_path, obj_viz_map)

    def save_viz_force(self, batch_idx, **data):
        i = self.cfg.eval_batch_size * batch_idx
        j = 0
        k = i*(self.accel.num_processes) + self.accel.process_index + j

        obj_vert = YCB_MESHES[data['obj_name'][j]]["verts_sampled"]
        obj_CoM = YCB_MESHES[data['obj_name'][j]]["CoM"]
        gt_obj_rt = data['gt_obj_rt'][j]
        pd_obj_rt = data['pd_obj_rt'][j]
        root_joint = data['root_joint'][j]

        gt_obj_vert = obj_vert @ gt_obj_rt[:3, :3].T + gt_obj_rt[:3, 3] - root_joint
        pd_obj_vert = obj_vert @ pd_obj_rt[:3, :3].T + pd_obj_rt[:3, 3] - root_joint
        gt_obj_CoM = obj_CoM @ gt_obj_rt[:3, :3].T + gt_obj_rt[:3, 3] - root_joint
        pd_obj_CoM = obj_CoM @ pd_obj_rt[:3, :3].T + pd_obj_rt[:3, 3] - root_joint

        gt_hand_vert = data['gt_hand_vert'][j]
        pd_hand_vert = data['pd_hand_vert'][j] - root_joint

        gravity = data['gravity'][j]

        gt_force_local = data['gt_force_local'][j]
        pd_force_local = data['pd_force_local'][j]

        if not data['is_right'][j]:
            gt_obj_vert[..., 0] = -gt_obj_vert[..., 0]
            pd_obj_vert[..., 0] = -pd_obj_vert[..., 0]
            gravity[..., 0] = -gravity[..., 0]
            pd_hand_vert[..., 0] = -pd_hand_vert[..., 0]
            gt_obj_CoM[..., 0] = -gt_obj_CoM[..., 0]
            pd_obj_CoM[..., 0] = -pd_obj_CoM[..., 0]
        
        gt_force_point, gt_force_global = from_local_to_global(gt_force_local, gt_hand_vert)
        pd_force_point, pd_force_global = from_local_to_global(pd_force_local, pd_hand_vert)

        gt_force = np.stack([gt_force_point, gt_force_point + gt_force_global*0.1], axis=1)
        pd_force = np.stack([pd_force_point, pd_force_point + pd_force_global*0.1], axis=1)

        gt_gravity = np.stack([gt_obj_CoM[None], gt_obj_CoM + gravity*0.3], axis=1)
        pd_gravity = np.stack([pd_obj_CoM[None], pd_obj_CoM + gravity*0.3], axis=1)
        
        force_dt = {
            'gt_obj_vert_#00FF00': gt_obj_vert, 
            'pd_obj_vert_#FF0000': pd_obj_vert, 
            'gt_force_line_#00FF00': gt_force,       
            'pd_force_line_#FF0000': pd_force, 
            'gt_hand_vert_#000000': gt_hand_vert, 
            'pd_hand_vert_#FF00FF': pd_hand_vert,
            'gt_gravity_line_#00FF00': gt_gravity,
            'pd_gravity_line_#FF0000': pd_gravity,
        }
        save_path = os.path.join(self.save_dir, f"viz/{k}_force.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f: pickle.dump(force_dt, f)

    def save_viz_mesh_force(self, batch_idx, **data):
        i = self.cfg.eval_batch_size * batch_idx
        j = 0
        k = i*(self.accel.num_processes) + self.accel.process_index + j

        # obj_vert = YCB_MESHES[data['obj_name'][j]]["verts_sampled"]
        obj_vert = np.array(YCB_MESHES[data['obj_name'][j]]["verts"])
        obj_face = np.array(YCB_MESHES[data['obj_name'][j]]["faces"])
        obj_CoM = np.array(YCB_MESHES[data['obj_name'][j]]["CoM"])
        gt_obj_rt = data['gt_obj_rt'][j]
        pd_obj_rt = data['pd_obj_rt'][j]
        root_joint = data['root_joint'][j]

        gt_obj_vert = obj_vert @ gt_obj_rt[:3, :3].T + gt_obj_rt[:3, 3] - root_joint
        pd_obj_vert = obj_vert @ pd_obj_rt[:3, :3].T + pd_obj_rt[:3, 3] - root_joint
        gt_obj_CoM = obj_CoM @ gt_obj_rt[:3, :3].T + gt_obj_rt[:3, 3] - root_joint
        pd_obj_CoM = obj_CoM @ pd_obj_rt[:3, :3].T + pd_obj_rt[:3, 3] - root_joint

        gt_hand_vert = data['gt_hand_vert'][j]
        pd_hand_vert = data['pd_hand_vert'][j] - root_joint
        hand_face = mano_layer_r.th_faces.clone().cpu().numpy()

        gravity = data['gravity'][j]

        gt_force_local = data['gt_force_local'][j]
        pd_force_local = data['pd_force_local'][j]

        if not data['is_right'][j]:
            gt_obj_vert[..., 0] = -gt_obj_vert[..., 0]
            pd_obj_vert[..., 0] = -pd_obj_vert[..., 0]
            gravity[..., 0] = -gravity[..., 0]
            pd_hand_vert[..., 0] = -pd_hand_vert[..., 0]
            gt_obj_CoM[..., 0] = -gt_obj_CoM[..., 0]
            pd_obj_CoM[..., 0] = -pd_obj_CoM[..., 0]
        
        gt_force_point, gt_force_global = from_local_to_global(gt_force_local, gt_hand_vert)
        pd_force_point, pd_force_global = from_local_to_global(pd_force_local, pd_hand_vert)

        gt_force = np.stack([gt_force_point, gt_force_point + gt_force_global*0.1], axis=1)
        pd_force = np.stack([pd_force_point, pd_force_point + pd_force_global*0.1], axis=1)

        gt_gravity = np.stack([gt_obj_CoM[None], gt_obj_CoM + gravity*0.1], axis=1)
        pd_gravity = np.stack([pd_obj_CoM[None], pd_obj_CoM + gravity*0.1], axis=1)

        rgb_path = data['rgb_path'][j]
        rgb = cv2.imread(rgb_path)

        K = data['K'][j]
        root = data['root_joint_flip'][j]
        
        gt_obj_mesh = {
            'vertex': gt_obj_vert + root,
            'face': obj_face,
        }
        pd_obj_mesh = {
            'vertex': pd_obj_vert + root,
            'face': obj_face,
        }
        pd_hand_mesh = {
            'vertex': pd_hand_vert + root,
            'face': hand_face,
        }
        gt_hand_mesh = {
            'vertex': gt_hand_vert + root,
            'face': hand_face,
        }

        force_dt = {
            'gt_mesh': gt_obj_mesh,
            'pd_mesh': pd_obj_mesh,
            'gt_force_line_#00FF00': gt_force + root,       
            'pd_force_line_#FF0000': pd_force + root, 
            'gt_hand_mesh': gt_hand_mesh,
            'pd_hand_mesh': pd_hand_mesh,
            'gt_gravity_line_#00FF00': gt_gravity + root,
            'pd_gravity_line_#FF0000': pd_gravity + root,
            'K': K,
        }
        save_path = os.path.join(self.save_dir, f"viz/{k}_force.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f: pickle.dump(force_dt, f)
        cv2.imwrite(os.path.join(self.save_dir, f"viz/{k}_rgb.jpg"), rgb)

def obj_dt_to_dataframe(obj_dt):
    for k, v in obj_dt.items():
        obj_dt[k] = dict(sorted(v.items()))
    df = pd.DataFrame(obj_dt)
    for k in ['MCE', 'MCE2', 'SMCE', 'OCE', 'ADD', 'ADDS']:
        if k in df:
            df[k] = df[k].map(lambda x: f"{1000*x:.2f}")
    for k in ['ADD01d', 'ADDS01d']:
        df[k] = df[k].map(lambda x: f"{x:.2%}")
    for k in ['REP']:
        df[k] = df[k].map(lambda x: f"{x:.2f}")
    return df
    
# modified from 2023_CVPR_HFL
def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    xyz_pred_list = [np.around(x, decimals=6).tolist() for x in xyz_pred_list]
    verts_pred_list = [np.around(x, decimals=6).tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))


import torch
import torch.nn as nn
from thop import profile
from copy import deepcopy

def compute_flops(model, batch):
    # if in ddp mode, return None
    if torch.distributed.is_initialized():
        return None
    
    example_input = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            example_input[k] = v[[0]].cuda()
        else:
            example_input[k] = [v[0]]
    class WrappedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = deepcopy(model).cuda().eval() 
        def forward(self, x):
            return self.model(x, mode='predict') 

    wrapped_model = WrappedModel(model)

    flops, params = profile(wrapped_model, inputs=(example_input,), verbose=False)

    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")


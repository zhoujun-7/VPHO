import os
import time
import torch
import random
import logging
import numpy as np
import pprint
from accelerate import Accelerator
from accelerate.logging import get_logger


from lib.configs.args import Config

def get_timestamp():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


class BaseTrainer:
    def __init__(self, cfg:Config):
        self.cfg = cfg

        self.accel = Accelerator(gradient_accumulation_steps=self.cfg.gradient_accumulation_steps)
        self.device = self.accel.device
        self.get_logger()

        if self.cfg.checkpoint:
            self.start_epoch = int(self.cfg.checkpoint.split('/')[-1].split('.')[0].split('epoch_')[-1])
        else:
            self.start_epoch = 0

        self.setup_seed()
        self.get_model()
        self.get_dataloader()
        self.get_optimizer()
        self.get_scheduler()
        self.prepare_accelerator()


    def setup_seed(self):
        seed = self.cfg.random_seed + self.accel.process_index * 1e8
        seed = int(seed)
        if seed == 0:
            torch.backends.cudnn.benchmark = True
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_logger(self):
        self.save_dir = os.path.join(self.cfg.output_dir, f"{get_timestamp()}_{self.cfg.mark}_{self.cfg.mode}_{self.cfg.model}")
        self.save_dir = self.accel.gather_for_metrics([self.save_dir], use_gather_object=True)[0] # to make sure the log file of each process is the same
        os.makedirs(self.save_dir, exist_ok=True)
        log_file = os.path.join(self.save_dir, 'info.log')
        logging.basicConfig(filename=log_file, format='%(asctime)-15s \n%(message)s')
        self.logger = get_logger(log_file)
        self.logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger("").addHandler(console)
        cfg_dt = pprint.pformat(vars(self.cfg), indent=4)
        self.logger.info(cfg_dt, main_process_only=True)
        

    def get_model(self):
        raise NotImplementedError
    
    def get_optimizer(self):
        raise NotImplementedError

    def get_scheduler(self):
        raise NotImplementedError

    def get_dataloader(self):
        raise NotImplementedError
    
    def prepare_accelerator(self):
        raise NotImplementedError

    def load_checkpoint(self):
        self.accel.load_state(self.cfg.checkpoint, strict=False)
        self.accel.wait_for_everyone()

    def save_checkpoint(self, epoch):
        self.accel.wait_for_everyone()
        save_path = os.path.join(self.save_dir, 'checkpoint', f'epoch_{epoch}.state')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.accel.save_state(save_path)

    def save_model(self):
        if self.accel.is_main_process:
            save_path = os.path.join(self.save_dir, 'final_model.pt')
            model = self.accel.unwrap_model(self.model)
            torch.save(model.state_dict(), save_path)
        self.accel.wait_for_everyone()
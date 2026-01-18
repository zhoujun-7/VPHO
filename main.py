import os

from lib.configs.args import cfg
# from lib.engine.train_hand import Trainer
from lib.engine.train_diff_hand_obj import Trainer


if __name__ == "__main__":
    trainer = Trainer(cfg)
    if cfg.mode == 'train':
        trainer.run()
    elif cfg.mode == 'infer_candidate':
        trainer.infer_candidate()
    elif cfg.mode == 'energy':
        raise NotImplementedError
    elif cfg.mode == 'eval':
        trainer.eval()
    elif cfg.mode == 'infer':
        if cfg.checkpoint:
            trainer.load_checkpoint()
        trainer.infer()
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")
    trainer.logger.info("All is done", main_process_only=True)
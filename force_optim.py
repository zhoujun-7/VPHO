import os

from lib.configs.args import cfg
from lib.engine.force_optimization import ForceOptimizer


if __name__ == "__main__":
    runner = ForceOptimizer(cfg)
    runner.optimize_batch()
import os.path as osp
from yacs.config import CfgNode as CN

from utils import common

_C = CN()
_C.project_name = 'Noisy-LM'
_C.exp_name = ''
_C.display_name = ''
_C.output_dir = ''
_C.seed = 42

# path params
_C.data = CN()
_C.data.name = 'OpenAIHumanFeedback'
_C.data.root_dir = ''
_C.data.types = []

_C.preprocess = CN()
_C.preprocess.noise_level = 0 # defined in percentage

_C.sft = CN()
_C.sft.max_len = 128

_C.train = CN()
_C.train.learning_rate = 1e-5
_C.train.batch_size = 16
_C.train.grad_acc_steps = 1
_C.train.max_epoch = 5
_C.train.eval_steps = 1000
_C.train.save_steps = 1000
_C.train.logging_steps = 50

_C.eval = CN()
_C.eval.batch_size = 1


def update_config(cfg, filename):
    cfg.defrost()
    cfg.merge_from_file(filename)
    
    cfg.output_dir = osp.join(cfg.output_dir, cfg.exp_name)
    common.ensure_dir(cfg.output_dir)
    cfg.freeze()
    
    return cfg
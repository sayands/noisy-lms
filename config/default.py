from yacs.config import CfgNode as CN

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

def update_config(cfg, filename):
    cfg.defrost()
    cfg.merge_from_file(filename)
    cfg.freeze()
    
    return cfg
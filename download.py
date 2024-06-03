import argparse
import os.path as osp
from datasets import load_dataset
import shutil

from config import update_config, config
from utils import common

def download(cfg):
    dataset_root_dir = osp.join(cfg.data.root_dir, cfg.data.name)
    dataset_dir = osp.join(dataset_root_dir, 'orig')
    
    if osp.exists(dataset_dir): 
        shutil.rmtree(dataset_dir)
    
    dataset_types = cfg.data.types
    for dataset_type in dataset_types:
        dataset = load_dataset("openai/summarize_from_feedback", dataset_type)
        save_path = osp.join(dataset_dir, dataset_type)
        common.ensure_dir(save_path)
        dataset.save_to_disk(save_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Download OpenAI Human Feedback dataset')
    parser.add_argument('--config', type=str, default='', help='Path to the config file')
    
    return parser.parse_known_args()

if __name__ == '__main__':
    args, _ = parse_args()
    cfg = update_config(config, args.config)
    
    download(cfg)
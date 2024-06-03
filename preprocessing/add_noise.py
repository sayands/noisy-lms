import os
import os.path as osp
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import sys
import logging as log

from datasets import Dataset, load_from_disk



from config import config, update_config
from utils import common

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
workspace_dir = os.environ['NOISY_LM_DIR']
src_dir = osp.join(workspace_dir, 'src')

sys.path.append(workspace_dir)
sys.path.append(src_dir)


class OpenAIHumanFeedbackNoise():
    def __init__(self, cfg, split) -> None:
        self.cfg = cfg
        self.split = split
        np.random.seed(self.cfg.seed)
        
        self.dataset_root_dir = osp.join(cfg.data.root_dir, cfg.data.name)    
        self.type = 'comparisons'
        self.noise_level = self.cfg.preprocess.noise_level
        
        assert self.noise_level < 20, "Noise Level too High, don't generate!"
        
        self.source_dataset_dir = osp.join(self.dataset_root_dir, 'orig', self.type)
        self.target_dataset_dir  = osp.join(self.dataset_root_dir, 'noise_{}'.format(str(self.noise_level).replace('.', '')), self.type)
        
        if osp.exists(self.target_dataset_dir): 
            shutil.rmtree(self.target_dataset_dir)
        
        common.ensure_dir(self.target_dataset_dir)

        self.load_dataset()
        log.info('Initialisation Complete')
        
    def load_dataset(self):
        self.dataset = load_from_disk(self.source_dataset_dir)
        self.dataset_num = len(self.dataset[self.split])
                
        log.info('Loaded dataset with length - {}, from - {}'.format(self.dataset_num, self.source_dataset_dir))

    def flip_labels(self):
        num_samples_to_add_noise = int((self.noise_level / 100.0) * self.dataset_num)
        noise_indices = np.random.choice(self.dataset_num, num_samples_to_add_noise, replace=False)
        
        log.info('Flipping labels for {} percent samples - {}'.format(self.noise_level, num_samples_to_add_noise))
        dataset_df = self.dataset[self.split].to_pandas()
        
        for index in tqdm(noise_indices):
            choice = self.dataset[self.split]["choice"][index]
            choice_noised = 1 if choice == 0 else 1
            dataset_df.at[index, "choice"] = choice_noised
        
        self.dataset[self.split] = Dataset.from_pandas(dataset_df)
            
        assert len(self.dataset[self.split]) == self.dataset_num
        log.info('Saving dataset to {}'.format(self.target_dataset_dir))
        self.dataset.save_to_disk(self.target_dataset_dir)
        

def parse_args():
    parser = argparse.ArgumentParser(description='Add Noise To dataset')
    parser.add_argument('--config', type=str, default='', help='Path to the config file')
    parser.add_argument('--split', type=str, default='train', help='split')
    
    return parser.parse_known_args()

if __name__ == '__main__':
    args, _ = parse_args()
    cfg = update_config(config, args.config)
    split = args.split
    
    openai_human_feedback = OpenAIHumanFeedbackNoise(cfg, args.split)
    openai_human_feedback.flip_labels()
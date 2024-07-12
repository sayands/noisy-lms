import numpy as np
from scipy.stats import entropy
import pandas as pd

import sys
import os
import os.path as osp
workspace_dir = os.environ['NOISY_LM_DIR']
src_dir = osp.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--lm_reward_score_csv_path', type=str, help='lm_reward_score_csv_path')
    parser.add_argument('--ref_model_reward_score_csv_path', type=str, help='ref_model_reward_score_csv_path')
    kl_config = parser.parse_args()
    
    lm_model_df = pd.read_csv(kl_config.lm_reward_score_csv_path)
    ref_model_df = pd.read_csv(kl_config.ref_model_reward_score_csv_path)

    lm_scores = lm_model_df['score'].astype(float).to_numpy()
    ref_scores = ref_model_df['score'].astype(float).to_numpy()

    D = entropy(lm_scores, ref_scores)
    print(D)
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

import sys
import os
import os.path as osp
workspace_dir = os.environ['NOISY_LM_DIR']
src_dir = osp.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)
import datasets
from src.datasets_process import build_dataset

from trl.commands.cli_utils import TrlParser
from src.configs import DatasetConfig, TokenizerConfig, RewardGeneratorConfig

device = 0 if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = TrlParser((RewardGeneratorConfig, DatasetConfig, TokenizerConfig))
    gen_config, dataset_config, tokenizer_config = parser.parse_args_and_config()
    np.random.seed(gen_config.sampling_seed)

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(gen_config.lm_model_name_or_path)
    reward_pipe = pipeline("text-classification", model=gen_config.reward_model_name_or_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    ref_model.cuda()

    dataset = build_dataset('nsampler', dataset_config.dataset_path, tokenizer, nsampler_config=gen_config)

    gen_kwargs = {"num_beams":gen_config.num_beams, 
                  "no_repeat_ngram_size":gen_config.no_repeat_ngram_size, 
                  "early_stopping": gen_config.early_stopping,
                  "do_sample": gen_config.do_sample,
                  "temperature": gen_config.temperature,
                  "top_k": gen_config.top_k, 
                  "top_p": gen_config.top_p, 
                  "pad_token_id": tokenizer.eos_token_id}
    sent_kwargs = {"top_k": None, "function_to_apply": "sigmoid"}

    output_length_sampler = LengthSampler(gen_config.output_min_length, gen_config.output_max_length)
    output_data = dict()
    dataset.set_format("pandas")
    df_batch = dataset[:].sample(gen_config.num_samples)
    output_data["query"] = df_batch["query"].tolist()
    query_tensors = df_batch["input_ids"].tolist()

    #response_tensors_ref = []
    response_tensors_best_of = []
    for i in tqdm(range(gen_config.num_samples), desc="Generating Rewards"):
        gen_len = output_length_sampler()

        query = torch.tensor(query_tensors[i])
        # generating copies of the same query for the Best-of-n sampling
        queries = query.repeat((gen_config.n_best_of, 1))
        output = ref_model.generate(queries.to(device), max_new_tokens=gen_len, **gen_kwargs).squeeze()
        response_tensors_best_of.append(tokenizer.batch_decode(output))

    scores_best_of = []
    for i, response in tqdm(enumerate(response_tensors_best_of), desc="Filtering Rewards"):
        scores_best_of.append(torch.tensor([output[0]["score"] for output in reward_pipe(response, **sent_kwargs)]))
        
    output_data["response"] = [
        response_tensors_best_of[i][a.argmax().item()] for i, a in enumerate(scores_best_of)
    ]
    output_data["score"] = [a.max().item() for a in scores_best_of]

    # store results in a dataframe
    df_results = pd.DataFrame(output_data)
    df_results.to_csv(gen_config.csv_save_dir)
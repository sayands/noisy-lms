import torch
import pandas as pd
import numpy as np
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
from src.configs import DatasetConfig, TokenizerConfig, NSamplerConfig

device = 0 if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = TrlParser((NSamplerConfig, DatasetConfig, TokenizerConfig))
    sampler_config, dataset_config, tokenizer_config = parser.parse_args_and_config()
    np.random.seed(sampler_config.sampling_seed)

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(sampler_config.lm_model_name_or_path)
    reward_pipe = pipeline("text-classification", model=sampler_config.reward_model_name_or_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    ref_model.cuda()

    dataset = build_dataset('nsampler', dataset_config.dataset_path, tokenizer, nsampler_config=sampler_config)

    gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}
    sent_kwargs = {"top_k": None, "function_to_apply": "none"}

    output_length_sampler = LengthSampler(sampler_config.output_min_length, sampler_config.output_max_length)
    output_data = dict()
    dataset.set_format("pandas")
    df_batch = dataset[:].sample(sampler_config.num_samples)
    output_data["query"] = df_batch["query"].tolist()
    query_tensors = df_batch["input_ids"].tolist()

    response_tensors_ref = []
    response_tensors_best_of = []
    for i in range(sampler_config.num_samples):
        gen_len = output_length_sampler()

        query = torch.tensor(query_tensors[i])

        output = ref_model.generate(query.unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs).squeeze()
        response_tensors_ref.append(tokenizer.decode(output))

        # generating copies of the same query for the Best-of-n sampling
        queries = query.repeat((sampler_config.n_best_of, 1))
        output = ref_model.generate(queries.to(device), max_new_tokens=gen_len, **gen_kwargs).squeeze()
        response_tensors_best_of.append(tokenizer.batch_decode(output))

    scores_ref = [output[0]["score"] for output in reward_pipe(response_tensors_ref, **sent_kwargs)]
    scores_best_of = []
    for i, response in enumerate(response_tensors_best_of):
        scores_best_of.append(torch.tensor([output[0]["score"] for output in reward_pipe(response, **sent_kwargs)]))
        
    output_data["response (ref)"] = response_tensors_ref
    output_data["scores (ref)"] = scores_ref
    output_data["response (best_of)"] = [
        response_tensors_best_of[i][a.argmax().item()] for i, a in enumerate(scores_best_of)
    ]
    output_data["scores (best_of)"] = [a.max().item() for a in scores_best_of]

    # store results in a dataframe
    df_results = pd.DataFrame(output_data)
    df_results.to_csv(sampler_config.csv_save_dir)
# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python $NOISY_LM_DIR/src/trainers/dpo_trainer_2.py \
    --dataset_name openai/summarize_from_feedback \
    --model_name_or_path gpt2 \
    --dataset_noise_level 0.05 \
    --dataset_noise_seed 42 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 50 \
    --eval_strategy steps \
    --eval_steps 500 \
    --warmup_steps 150 \
    --report_to wandb \
    --logging_first_step \
    --no_remove_unused_columns \
    --output_dir /home/anghosh/PROJECT/output \
    --run_name dpo_trainer_wnoise \
"""

import logging
import os

from trl.commands.cli_utils import DPOScriptArguments, TrlParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import sys
import os
import os.path as osp
workspace_dir = os.environ['NOISY_LM_DIR']
src_dir = osp.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)

from src.configs import DatasetConfig, TokenizerConfig
from src.datasets_process import build_dataset


if __name__ == "__main__":
    logging.basicConfig(filename='dpo_trl.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig, DatasetConfig))
    logging.info("START\n")
    # print("parser\n", dir(parser), "\n", vars(parser))
    args, training_args, model_config, dataset_config = parser.parse_args_and_config()
    # print("args", dir(args), "\n", vars(args))
    # print("training_args", dir(training_args), "\n", vars(training_args))
    # print("model_config", dir(model_config), "\n", vars(model_config))

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]


    ################
    # Dataset
    ################
    dataset = build_dataset('DPO', args.dataset_name, tokenizer, dataset_config.dataset_noise_level, dataset_config.dataset_noise_seed)
    dataset = dataset.select_columns(['prompt', 'chosen', 'rejected'])

    #logging statements for sanity check
    keys = dataset["train"]
    logging.info(keys)
    logging.info(keys['prompt'][1])
    logging.info(keys['chosen'][1])
    logging.info(keys['rejected'][1])
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    logging.info(train_dataset)
    logging.info(train_dataset['prompt'][1])
    logging.info(train_dataset['chosen'][1])
    logging.info(train_dataset['rejected'][1])

    ################
    # Training
    ################

    trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
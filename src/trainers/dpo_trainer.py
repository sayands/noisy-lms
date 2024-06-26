"""
# regular:
python $NOISY_LM_DIR/src/trainers/dpo_trainer.py \
    --dataset_path openai/summarize_from_feedback \
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
from src.utils import common

if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, TokenizerConfig, DPOConfig, ModelConfig, DatasetConfig))
    args, tokenizer_config, training_args, model_config, dataset_config = parser.parse_args_and_config()

    dataset_noise_level_name = str(dataset_config.dataset_noise_level).split('.')
    dataset_noise_level_name = str(dataset_noise_level_name[0]) + str(dataset_noise_level_name[1])
    
    training_args.run_name =  training_args.run_name + '_' + dataset_noise_level_name
    training_args.output_dir = osp.join(training_args.output_dir, training_args.run_name)
    
    common.ensure_dir(training_args.output_dir)

    # Model & Tokenizer
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

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Dataset
    dataset = build_dataset('DPO', dataset_config.dataset_path, tokenizer, dataset_config.dataset_noise_level, dataset_config.dataset_noise_seed)
    dataset = dataset.select_columns(['prompt', 'chosen', 'rejected'])

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # Training
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
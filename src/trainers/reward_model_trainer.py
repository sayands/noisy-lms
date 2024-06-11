import os
import os.path as osp
import logging

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments
)
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    get_kbit_device_map,
    get_quantization_config,
)

import sys
workspace_dir = os.environ['NOISY_LM_DIR']
src_dir = osp.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)

from src.configs import DatasetConfig
from src.datasets import make_dataset
from src.utils import common

if __name__ == "__main__":
    parser = HfArgumentParser((ModelConfig, DatasetConfig, RewardConfig))  # type: ignore[reportArgumentType]
    model_config, dataset_config, reward_config = parser.parse_args_into_dataclasses()
    
    dataset_noise_level_name = str(dataset_config.dataset_noise_level).split('.')
    dataset_noise_level_name = str(dataset_noise_level_name[0]) + str(dataset_noise_level_name[1])
    
    reward_config.run_name =  reward_config.run_name + '_' + dataset_noise_level_name
    reward_config.output_dir = osp.join(reward_config.output_dir, reward_config.run_name)
    common.ensure_dir(reward_config.output_dir)

    if (
        dataset_config.preprocess_for_reward_trainer
        and not dataset_config.preprocess_tokenizer
    ):
        # TODO: Handle local models, this will crash otherwise
        dataset_config.preprocess_tokenizer = model_config.model_name_or_path

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    

    dataset_config.max_token_length = reward_config.max_length
    dataset = make_dataset(dataset_config)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    
    reward_config = RewardConfig(
        output_dir=reward_config.output_dir,
        overwrite_output_dir = True,
        eval_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate= reward_config.learning_rate,
        per_device_train_batch_size= reward_config.per_device_train_batch_size,
        per_device_eval_batch_size= reward_config.per_device_eval_batch_size,
        half_precision_backend=True,
        fp16=True,
        max_length=reward_config.max_length,
        adam_beta1=reward_config.adam_beta1, adam_beta2=reward_config.adam_beta2,
        gradient_accumulation_steps= reward_config.gradient_accumulation_steps,
        num_train_epochs=reward_config.num_train_epochs,
        warmup_steps=reward_config.warmup_steps,
        eval_steps=reward_config.eval_steps,
        save_steps=reward_config.save_steps,
        load_best_model_at_end=True,
        logging_steps=reward_config.logging_steps,
        remove_unused_columns=False,
        run_name = reward_config.run_name)

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    trainer.save_model(reward_config.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    logging.info(metrics)

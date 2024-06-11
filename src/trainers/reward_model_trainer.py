""" 
Usage:

python3.9 train_reward_model.py \
    --output_dir=out \
    --model_name_or_path=facebook/opt-350m \
    --max_length=128 \
    --dataset_name=openai/summarize_from_feedback \
    --dataset_noise_type=flip_labels \
    --dataset_noise_level=0.5 \
    --dataset_noise_seed=42 \
    --preprocess_for_reward_trainer \
    --preprocess_tokenizer=facebook/opt-350m \
    --report_to=wandb \
    --run_name=fb-opt-350m-0.5-noise 
"""

import logging

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    get_kbit_device_map,
    get_quantization_config,
)

from src.datasets.common import DatasetConfig

if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig, DatasetConfig))  # type: ignore[reportArgumentType]
    reward_config, model_config, dataset_config = parser.parse_args_into_dataclasses()
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

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

    dataset = dataset_config.make_dataset()
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

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

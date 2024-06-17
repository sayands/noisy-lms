import os
import os.path as osp

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl.trainer.ppov2_config import PPOv2Config
from trl.trainer.ppov2_trainer import PPOv2Trainer
from trl import (
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.commands.cli_utils import TrlParser
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

import sys

workspace_dir = os.environ["NOISY_LM_DIR"]
src_dir = osp.join(workspace_dir, "src")
sys.path.append(workspace_dir)
sys.path.append(src_dir)

from src.configs import DatasetConfig, TokenizerConfig
from src.datasets import make_dataset
from src.utils import common

if __name__ == "__main__":
    parser = TrlParser((PPOv2Config, TokenizerConfig, ModelConfig, DatasetConfig))
    train_config, tokenizer_config, model_config, dataset_config = (
        parser.parse_args_and_config()
    )

    dataset_noise_level_name = str(dataset_config.dataset_noise_level).split(".")
    dataset_noise_level_name = str(dataset_noise_level_name[0]) + str(
        dataset_noise_level_name[1]
    )

    train_config.run_name = train_config.run_name + "_" + dataset_noise_level_name
    train_config.output_dir = osp.join(train_config.output_dir, train_config.run_name)

    common.ensure_dir(train_config.output_dir)

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
        use_cache=False if train_config.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        train_config.reward_model_path,
        num_labels=1,
        **model_kwargs
    )

    value_model = AutoModelForSequenceClassification.from_pretrained(
        train_config.reward_model_path,
        num_labels=1,
        **model_kwargs
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )

    model_ref = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_path, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE


    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    dataset_config.max_token_length = 128 #train_config.max_length
    dataset_config.preprocess_ppo_tokenizer = tokenizer

    dataset = make_dataset(dataset_config)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    num_episodes = int(len(train_dataset) * train_config.num_train_epochs)

    training_args = PPOv2Config(
        output_dir=train_config.output_dir,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate=train_config.learning_rate,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        half_precision_backend="auto",
        fp16=True,
        adam_beta1=train_config.adam_beta1,
        total_episodes = num_episodes,
        adam_beta2=train_config.adam_beta2,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        num_train_epochs=train_config.num_train_epochs,
        warmup_steps=train_config.warmup_steps,
        eval_steps=train_config.eval_steps,
        save_steps=train_config.save_steps,
        load_best_model_at_end=True,
        logging_steps=train_config.logging_steps,
        remove_unused_columns=False,
        run_name=train_config.run_name,
        report_to="tensorboard",
    )

    trainer = PPOv2Trainer(
        config=training_args,
        tokenizer=tokenizer,
        policy=model,
        ref_policy=model_ref,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

    trainer.state.log_history()

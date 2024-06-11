import os
import os.path as osp

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
from trl.commands.cli_utils import TrlParser

import sys
workspace_dir = os.environ['NOISY_LM_DIR']
src_dir = osp.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)

from src.configs import DatasetConfig, TokenizerConfig
from src.datasets import make_dataset
from src.utils import common

if __name__ == "__main__":
    parser = TrlParser((DPOConfig, TokenizerConfig, ModelConfig, DatasetConfig))
    train_config, tokenizer_config, model_config, dataset_config = parser.parse_args_and_config()
    
    dataset_noise_level_name = str(dataset_config.dataset_noise_level).split('.')
    dataset_noise_level_name = str(dataset_noise_level_name[0]) + str(dataset_noise_level_name[1])
    
    train_config.run_name =  train_config.run_name + '_' + dataset_noise_level_name
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
    
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    
    model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    
    dataset_config.max_token_length = train_config.max_length
    dataset = make_dataset(dataset_config)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    
    training_args = DPOConfig(
        output_dir=train_config.output_dir,
        overwrite_output_dir = True,
        eval_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate= train_config.learning_rate,
        per_device_train_batch_size= train_config.per_device_train_batch_size,
        per_device_eval_batch_size= train_config.per_device_eval_batch_size,
        half_precision_backend="auto",
        fp16=True,
        max_prompt_length=train_config.max_length,
        max_length=train_config.max_length,
        adam_beta1=train_config.adam_beta1, adam_beta2=train_config.adam_beta2,
        gradient_accumulation_steps= train_config.gradient_accumulation_steps,
        num_train_epochs=train_config.num_train_epochs,
        warmup_steps=train_config.warmup_steps,
        eval_steps=train_config.eval_steps,
        save_steps=train_config.save_steps,
        load_best_model_at_end=True,
        logging_steps=train_config.logging_steps,
        remove_unused_columns=False,
        loss_type= "sigmoid",
        run_name = train_config.run_name)
    
    trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    
    
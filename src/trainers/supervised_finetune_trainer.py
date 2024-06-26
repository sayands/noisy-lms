import os
import os.path as osp

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    HfArgumentParser
)

from trl import RewardConfig

import evaluate

import sys
workspace_dir = os.environ['NOISY_LM_DIR']
src_dir = osp.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)

from src.datasets_process import tldr
from configs import DatasetConfig
from utils import common

if __name__ == '__main__':
    parser = HfArgumentParser((DatasetConfig, RewardConfig))  # type: ignore[reportArgumentType]
    dataset_config, reward_config = parser.parse_args_into_dataclasses()
    
    reward_config.output_dir = osp.join(reward_config.output_dir, reward_config.run_name)
    common.ensure_dir(reward_config.output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    
    data_path = dataset_config.dataset_path
    train_dataset = tldr.TLDRDataset(data_path, tokenizer, "train", max_length=reward_config.max_length)    
    eval_dataset = tldr.TLDRDataset(data_path, tokenizer, "valid", max_length=reward_config.max_length)
    
    rouge = evaluate.load("rouge")
    
    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = rouge.compute(predictions=pred_str, references=label_str)
        return result

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    training_args = TrainingArguments(
        output_dir=reward_config.output_dir,
        overwrite_output_dir = True,
        eval_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate= reward_config.learning_rate,
        per_device_train_batch_size= reward_config.per_device_train_batch_size,
        per_device_eval_batch_size= reward_config.per_device_eval_batch_size,
        half_precision_backend="auto",
        fp16=True,
        adam_beta1=reward_config.adam_beta1, adam_beta2=reward_config.adam_beta2,
        gradient_accumulation_steps= reward_config.gradient_accumulation_steps,
        num_train_epochs=reward_config.num_train_epochs,
        warmup_steps=reward_config.warmup_steps,
        eval_steps=reward_config.eval_steps,
        save_steps=reward_config.save_steps,
        load_best_model_at_end=True,
        logging_steps=reward_config.logging_steps,
        run_name = reward_config.run_name)
    
    trainer = Trainer(
        model=model,args=training_args, train_dataset=train_dataset, 
        eval_dataset=eval_dataset, compute_metrics=compute_metrics,
        data_collator=default_data_collator, preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        
    trainer.train()
    trainer.save_model(reward_config.output_dir)
    
    
    
    
    
    
    
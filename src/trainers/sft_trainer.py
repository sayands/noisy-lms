import os
import os.path as osp
import argparse
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import evaluate

import sys
workspace_dir = os.environ['NOISY_LM_DIR']
src_dir = osp.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)

from src.datasets import tldr
from config import config, update_config

rouge = evaluate.load("rouge")

class SFTModelTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.out_dir = cfg.output_dir
        
        self.data_path = "CarperAI/openai_summarize_tldr"    # download dataset on the fly
        self.max_input_len = cfg.sft.max_len
        
        self.registerModelAndTokenizer()
        self.registerDataset()
        
        # Set up the metric
        
        
        self.training_args = TrainingArguments(
        output_dir=self.out_dir,
        eval_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate= cfg.train.learning_rate,
        per_device_train_batch_size= cfg.train.batch_size,
        per_device_eval_batch_size= cfg.eval.batch_size,
        gradient_checkpointing=True,
        half_precision_backend=True,
        fp16=True,
        adam_beta1=0.9, adam_beta2=0.95,
        gradient_accumulation_steps= cfg.train.grad_acc_steps,
        num_train_epochs=cfg.train.max_epoch,
        warmup_steps=100,
        eval_steps=cfg.train.eval_steps,
        save_steps=cfg.train.save_steps,
        load_best_model_at_end=True,
        logging_steps=cfg.train.logging_steps,
        run_name = cfg.exp_name
    )

    def compute_metrics(self, eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = self.rouge.compute(predictions=pred_str, references=label_str)
        return result

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    def registerModelAndTokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2", use_cache=False)
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        
        self.model = model
        self.tokenizer = tokenizer
    
    def registerDataset(self):
        train_dataset = tldr.TLDRDataset(
        self.data_path, self.tokenizer,
        "train", max_length=self.max_input_len)    
        
        dev_dataset = tldr.TLDRDataset(
        self.data_path, self.tokenizer,
        "valid", max_length=self.max_input_len)
        
        self.train_dataset = train_dataset
        self.val_dataset = dev_dataset
    
    def train_model(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=default_data_collator,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics)
        
        trainer.train()
        trainer.save_model(self.out_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Add Noise To dataset')
    parser.add_argument('--config', type=str, default='', help='Path to the config file')
    
    return parser.parse_known_args()

if __name__ == '__main__':
    args, _ = parse_args()
    cfg = update_config(config, args.config)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    
    data_path = "CarperAI/openai_summarize_tldr" 
    train_dataset = tldr.TLDRDataset(data_path, tokenizer, "train", max_length=cfg.sft.max_len)    
    eval_dataset = tldr.TLDRDataset(data_path, tokenizer, "valid", max_length=cfg.sft.max_len)
    
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
        output_dir=cfg.output_dir,
        overwrite_output_dir = True,
        eval_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate= cfg.train.learning_rate,
        per_device_train_batch_size= cfg.train.batch_size,
        per_device_eval_batch_size= cfg.eval.batch_size,
        half_precision_backend=True,
        fp16=True,
        adam_beta1=0.9, adam_beta2=0.95,
        gradient_accumulation_steps= cfg.train.grad_acc_steps,
        num_train_epochs=cfg.train.max_epoch,
        warmup_steps=100,
        eval_steps=cfg.train.eval_steps,
        save_steps=cfg.train.save_steps,
        load_best_model_at_end=True,
        logging_steps=cfg.train.logging_steps,
        run_name = cfg.exp_name)
    
    trainer = Trainer(
        model=model,args=training_args, train_dataset=train_dataset, 
        eval_dataset=eval_dataset, compute_metrics=compute_metrics,
        data_collator=default_data_collator, preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        
    trainer.train()
    trainer.save_model(cfg.output_dir)
    
    
    
    
    
    
    
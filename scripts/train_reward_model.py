# Usage: python scripts/train_reward_model.py --output_dir ./ --model_name_or_path=facebook/opt-350m --max_length=128
from trl import (
    RewardTrainer,
    ModelConfig,
    RewardConfig,
    get_kbit_device_map,
    get_quantization_config,
)
import torch
import datasets
from transformers import (
    AutoModelForSequenceClassification,
    HfArgumentParser,
    AutoTokenizer,
)
import logging

if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig))  # type: ignore[reportArgumentType]
    config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

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

    if model_config.lora_task_type != "SEQ_CLS":
        logging.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    # HARDCODED <3
    dataset = datasets.load_from_disk(
        "/media/sdebsarkar/extra-hdd/datasets/CSNLP/OpenAIHumanFeedback/orig/comparisons"
    )
    
    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for summary, choice in zip(examples['summaries'], examples['choice']):
            chosen = summary[choice]["text"]
            rejected = summary[abs(choice - 1)]["text"]

            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])

            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(
                tokenized_rejected["attention_mask"]
            )

        return new_examples

    dataset = dataset.map(preprocess_function, batched = True, num_proc=4)
    dataset = dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= 512 and len(x["input_ids_rejected"]) <= 512
    )
    
    train_dataset = dataset["train"]
    train_dataset = train_dataset.remove_columns(
        ["info", "summaries", "choice", "worker", "batch", "split", "extra"]
    )
    val_dataset = dataset["validation"]
    val_dataset = val_dataset.remove_columns(
        ["info", "summaries", "choice", "worker", "batch", "split", "extra"]
    )

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    logging.info(metrics)
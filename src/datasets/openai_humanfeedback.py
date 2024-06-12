import numpy as np
import functools
from typing import Any, Protocol, Union

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from src.configs import DatasetConfig
from .base import DatasetNoisifier, DatasetRewardTrainerPreprocessor, ConstructFromConfig

class OpenAIHumanFeedbackDatasetPreprocessor(
    DatasetRewardTrainerPreprocessor,
    DatasetNoisifier,
    ConstructFromConfig,
):
    DATASET_URL: str = "openai/summarize_from_feedback"

    @classmethod
    def _flip_labels(
        cls,
        dataset: datasets.Dataset,
        cfg: DatasetConfig,
    ) -> datasets.Dataset:
        num_samples_to_add_noise = int(cfg.dataset_noise_level * len(dataset))
        
        np.random.seed(cfg.dataset_noise_seed)
        noise_indices = np.random.choice(
            len(dataset), num_samples_to_add_noise, replace=False
        )

        def flip_label(example, idx):
            if idx in noise_indices:
                example["choice"] = 1 - example["choice"]
            return example

        return dataset.map(flip_label, with_indices=True)

    @classmethod
    def add_noise(
        cls,
        dataset: datasets.Dataset,
        cfg: DatasetConfig,
    ) -> datasets.Dataset:
        if cfg.dataset_noise_type == "flip_labels":
            return cls._flip_labels(dataset, cfg)
        else:
            raise ValueError(f"Unknown noise type: {cfg.dataset_noise_type}")

    @classmethod
    def preprocess_batch_for_dpo(
        cls,
        examples: dict[str, Any],
    ) -> datasets.Dataset:
        new_examples = { "prompt" : [], "chosen" :  [], "rejected" : []}
        
        for info, summary, choice in zip(examples["info"], examples["summaries"], examples["choice"]):
            prompt = info["post"]
            chosen = summary[choice]["text"]
            rejected = summary[1 - choice]["text"]
            
            new_examples["prompt"].append(prompt)
            new_examples["chosen"].append(chosen)
            new_examples["rejected"].append(rejected)
        return new_examples
    
    @classmethod
    def preprocess_batch_for_reward_trainer(
        cls,
        examples: dict[str, Any],
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
        max_length: int = 128
    ) -> dict[str, Any]:
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }

        for summary, choice in zip(examples["summaries"], examples["choice"]):
            chosen = summary[choice]["text"]
            rejected = summary[1 - choice]["text"]

            tokenized_chosen = tokenizer(chosen, truncation=True)
            tokenized_rejected = tokenizer(rejected, truncation=True)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(
                tokenized_chosen["attention_mask"]
            )

            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(
                tokenized_rejected["attention_mask"]
            )

        return new_examples

    @classmethod
    def from_config(
        cls,
        cfg: DatasetConfig
    ) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset(cls.DATASET_URL, "comparisons")
        assert isinstance(dataset_dict, datasets.DatasetDict)

        dataset_dict["validation"] = dataset_dict["validation"].select(range(2000))
        processed_train = cls.add_noise(dataset_dict["train"], cfg)
        processed_validation = cls.add_noise(dataset_dict["validation"], cfg)

        if cfg.preprocess_for_reward_trainer:
            assert cfg.preprocess_tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.preprocess_tokenizer, use_fast=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
            processed_train = processed_train.map(
                functools.partial(
                    cls.preprocess_batch_for_reward_trainer,
                    tokenizer=tokenizer,
                    max_length=cfg.max_token_length
                ),
                batched=True,
                num_proc=4,
            )
            processed_validation = processed_validation.map(
                functools.partial(
                    cls.preprocess_batch_for_reward_trainer,
                    tokenizer=tokenizer,
                    max_length=cfg.max_token_length
                ),
                batched=True,
                num_proc=4,
            )
            
            processed_train = processed_train.filter(
                lambda x: len(x["input_ids_chosen"]) <= cfg.max_token_length and len(x["input_ids_rejected"]) <= cfg.max_token_length
            )
            processed_validation = processed_validation.filter(
                lambda x: len(x["input_ids_chosen"]) <= cfg.max_token_length and len(x["input_ids_rejected"]) <= cfg.max_token_length
            )
            
        elif cfg.preprocess_for_dpo:
            processed_train = processed_train.map(
                functools.partial(
                    cls.preprocess_batch_for_dpo,
                ),
                batched=True,
                num_proc=4,
            )
            processed_validation = processed_validation.map(
                functools.partial(
                    cls.preprocess_batch_for_dpo,
                ),
                batched=True,
                num_proc=4,
            )
        else:
            raise NotImplementedError

        return datasets.DatasetDict(
            {
                "train": processed_train,
                "validation": processed_validation,
            }
        )
import numpy as np
import functools
from typing import Any, Protocol, Union
import torch

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from trl.core import LengthSampler

from src.configs import DatasetConfig
from .base import (
    DatasetNoisifier,
    DatasetRewardTrainerPreprocessor,
    ConstructFromConfig,
)


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
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
    ) -> dict[str, Any]:
        new_examples = {"prompt": [], "chosen": [], "rejected": []}

        for info, summary, choice in zip(
            examples["info"], examples["summaries"], examples["choice"]
        ):
            prompt = info["post"]
            chosen = summary[choice]["text"]
            rejected = summary[1 - choice]["text"]

            chosen = [
                {"content": prompt, "role": "user"},
                {"content": chosen, "role": "assistant"},
            ]
            rejected = [
                {"content": prompt, "role": "user"},
                {"content": rejected, "role": "assistant"},
            ]

            chosen = tokenizer.apply_chat_template(chosen, tokenize=False)
            rejected = tokenizer.apply_chat_template(rejected, tokenize=False)

            new_examples["prompt"].append(prompt)
            new_examples["chosen"].append(chosen)
            new_examples["rejected"].append(rejected)
        return new_examples

    @classmethod
    def preprocess_batch_for_ppo(
        cls,
        examples: dict[str, Any],
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
    ) -> dict[str, Any]:
        new_examples = {"input_ids": [], "lengths": []}

        for info in examples["info"]:
            chat_template_dict = [{"content": info["post"], "role": "user"}]
            input_ids = tokenizer.apply_chat_template(
                chat_template_dict,
                padding=False,
                add_generation_prompt=True,
            )
            new_examples["input_ids"].append(input_ids)
            new_examples["lengths"].append(len(input_ids))

        return new_examples

    @classmethod
    def preprocess_batch_for_kto(
        cls,
        examples: dict[str, Any],
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
    ) -> dict[str, Any]:
        new_examples = {"prompt": [], "completion": [], "label": []}

        for info, summary, choice in zip(
            examples["info"], examples["summaries"], examples["choice"]
        ):
            prompt = info["post"]
            chosen = summary[choice]["text"]
            rejected = summary[1 - choice]["text"]

            chosen = [{"content": chosen, "role": "assistant"}]
            rejected = [{"content": rejected, "role": "assistant"}]

            chosen = tokenizer.apply_chat_template(chosen, tokenize=False)
            rejected = tokenizer.apply_chat_template(rejected, tokenize=False)
            prompt = tokenizer.apply_chat_template(
                [{"content": prompt, "role": "user"}], tokenize=False
            )

            new_examples["prompt"].append(prompt)
            new_examples["completion"].append(chosen)
            new_examples["label"].append(True)

            new_examples["prompt"].append(prompt)
            new_examples["completion"].append(rejected)
            new_examples["label"].append(False)

        return new_examples

    @classmethod
    def preprocess_batch_for_reward_trainer(
        cls,
        examples: dict[str, Any],
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
        max_length: int = 128,
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
    def dpo_build(
        cls, dataset_path: str, tokenizer, dataset_noise_level, dataset_noise_seed
    ) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset(dataset_path, "comparisons")
        assert isinstance(dataset_dict, datasets.DatasetDict)

        def flip_label(example, idx):
            if idx in noise_indices:
                example["choice"] = 1 - example["choice"]
            return example

        def preprocess_batch_for_dpo(
            examples: dict[str, Any],
        ) -> dict[str, Any]:
            new_examples = {"prompt": [], "chosen": [], "rejected": []}

            for info, summary, choice in zip(
                examples["info"], examples["summaries"], examples["choice"]
            ):
                prompt = info["post"]
                chosen = summary[choice]["text"]
                rejected = summary[1 - choice]["text"]

                chosen = [
                    {"content": prompt, "role": "user"},
                    {"content": chosen, "role": "assistant"},
                ]
                rejected = [
                    {"content": prompt, "role": "user"},
                    {"content": rejected, "role": "assistant"},
                ]

                chosen = tokenizer.apply_chat_template(chosen, tokenize=False)
                rejected = tokenizer.apply_chat_template(rejected, tokenize=False)

                new_examples["prompt"].append(prompt)
                new_examples["chosen"].append(chosen)
                new_examples["rejected"].append(rejected)
            return new_examples

        # adding noise to dataset train split
        processed_train = dataset_dict["train"]
        num_samples_to_add_noise = int(dataset_noise_level * len(processed_train))
        np.random.seed(dataset_noise_seed)
        noise_indices = np.random.choice(
            len(processed_train), num_samples_to_add_noise, replace=False
        )
        processed_train = processed_train.map(flip_label, with_indices=True)
        # selecting 2000 samples from validation split
        dataset_dict["validation"] = dataset_dict["validation"].select(range(2000))
        processed_validation = dataset_dict["validation"]

        processed_train = processed_train.map(
            preprocess_batch_for_dpo,
            batched=True,
            num_proc=4,
        )
        processed_validation = processed_validation.map(
            preprocess_batch_for_dpo,
            batched=True,
            num_proc=4,
        )

        return datasets.DatasetDict(
            {
                "train": processed_train,
                "validation": processed_validation,
            }
        )

    @classmethod
    def ppo_build(cls, 
                  dataset_name,
                  tokenizer
                  ) -> torch.utils.data.Dataset:
    
        dataset_dict = datasets.load_dataset(dataset_name, "comparisons")
        assert isinstance(dataset_dict, datasets.DatasetDict)

        def preprocess_batch_for_ppo(examples):
            examples["input_ids"] = tokenizer.encode(examples["info"]["post"])
            examples["query"] = tokenizer.decode(examples["input_ids"])
            examples["lengths"] = len(examples["input_ids"])
            return examples

        processed_train = dataset_dict["train"]

        processed_train = processed_train.map(
            preprocess_batch_for_ppo,
            batched=False,
            )
        processed_train.select_columns(["input_ids", "query", "lengths"])
        processed_train.set_format(type='torch')
        
        return processed_train

    @classmethod
    def nsampler_build(cls, 
                       dataset_name, 
                       tokenizer, 
                       nsampler_config
                       )-> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset(dataset_name, "comparisons")
        assert isinstance(dataset_dict, datasets.DatasetDict)
        
        dataset_dict["validation"] = dataset_dict["validation"].select(range(2000))
        processed_validation = dataset_dict["validation"]

        if nsampler_config.sample_input_length:
            input_size = LengthSampler(nsampler_config.input_min_text_length, nsampler_config.input_max_text_length)

        def tokenize(sample):
            if nsampler_config.sample_input_length:
                sample["input_ids"] = tokenizer.encode(sample["info"]["post"])[: input_size()]
            else:
                sample["input_ids"] = tokenizer.encode(sample["info"]["post"])
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        processed_validation = processed_validation.map(tokenize, batched=False)
        processed_validation.set_format(type="torch")
        return processed_validation

    @classmethod
    def from_config(cls, cfg: DatasetConfig) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset(cls.DATASET_URL, "comparisons")
        assert isinstance(dataset_dict, datasets.DatasetDict)

        dataset_dict["validation"] = dataset_dict["validation"].select(range(2000))
        processed_train = cls.add_noise(dataset_dict["train"], cfg)
        processed_validation = dataset_dict["validation"]

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
                    max_length=cfg.max_token_length,
                ),
                batched=True,
                num_proc=4,
            )
            processed_validation = processed_validation.map(
                functools.partial(
                    cls.preprocess_batch_for_reward_trainer,
                    tokenizer=tokenizer,
                    max_length=cfg.max_token_length,
                ),
                batched=True,
                num_proc=4,
            )

            processed_train = processed_train.filter(
                lambda x: len(x["input_ids_chosen"]) <= cfg.max_token_length
                and len(x["input_ids_rejected"]) <= cfg.max_token_length
            )
            processed_validation = processed_validation.filter(
                lambda x: len(x["input_ids_chosen"]) <= cfg.max_token_length
                and len(x["input_ids_rejected"]) <= cfg.max_token_length
            )

        elif cfg.preprocess_for_dpo:
            tokenizer = cfg.preprocess_dpo_tokenizer
            processed_train = processed_train.map(
                functools.partial(
                    cls.preprocess_batch_for_dpo,
                    tokenizer=tokenizer,
                ),
                batched=True,
                num_proc=4,
            )
            processed_validation = processed_validation.map(
                functools.partial(
                    cls.preprocess_batch_for_dpo,
                    tokenizer=tokenizer,
                ),
                batched=True,
                num_proc=4,
            )
        elif cfg.preprocess_for_ppo:
            tokenizer = cfg.preprocess_ppo_tokenizer
            processed_train = processed_train.map(
                functools.partial(
                    cls.preprocess_batch_for_ppo,
                    tokenizer=tokenizer,
                ),
                remove_columns=processed_train.column_names,
                batched=True,
                num_proc=4,
            )
            processed_validation = processed_validation.map(
                functools.partial(
                    cls.preprocess_batch_for_ppo,
                    tokenizer=tokenizer,
                ),
                remove_columns=processed_validation.column_names,
                batched=True,
                num_proc=4,
            )

            processed_train = processed_train.filter(
                lambda x: x["lengths"] <= cfg.max_token_length
            )
            processed_validation = processed_validation.filter(
                lambda x: x["lengths"] <= cfg.max_token_length
            )

        elif cfg.preprocess_for_kto:
            tokenizer = cfg.preprocess_kto_tokenizer
            processed_train = processed_train.map(
                functools.partial(
                    cls.preprocess_batch_for_kto,
                    tokenizer=tokenizer,
                ),
                remove_columns=processed_train.column_names,
                batched=True,
                num_proc=4,
            )
            processed_validation = processed_validation.map(
                functools.partial(
                    cls.preprocess_batch_for_kto,
                    tokenizer=tokenizer,
                ),
                remove_columns=processed_validation.column_names,
                batched=True,
                num_proc=4,
            )

        elif cfg.preprocess_for_rloo:
            tokenizer = cfg.preprocess_rloo_tokenizer

            # Same as PPO?
            processed_train = processed_train.map(
                functools.partial(
                    cls.preprocess_batch_for_ppo,
                    tokenizer=tokenizer,
                ),
                remove_columns=processed_train.column_names,
                batched=True,
                num_proc=4,
            )

            # Same as PPO?
            processed_validation = processed_validation.map(
                functools.partial(
                    cls.preprocess_batch_for_ppo,
                    tokenizer=tokenizer,
                ),
                remove_columns=processed_validation.column_names,
                batched=True,
                num_proc=4,
            )

        return datasets.DatasetDict(
            {
                "train": processed_train,
                "validation": processed_validation,
            }
        )

import functools
from dataclasses import dataclass
from typing import Any, Protocol, Union

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

import datasets


@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_noise_type: str = "flip_labels"
    dataset_noise_level: float = 0.0
    dataset_noise_seed: int = 42
    preprocess_for_reward_trainer: bool = False
    preprocess_tokenizer: Union[str, None] = None

    def make_dataset(self) -> datasets.DatasetDict:
        return DATASET_NAME_TO_PREPROCESSOR[self.dataset_name].from_config(self)


class DatasetNoisifier(Protocol):
    @classmethod
    def add_noise(
        cls, dataset: datasets.Dataset, cfg: DatasetConfig
    ) -> datasets.Dataset: ...


class DatasetRewardTrainerPreprocessor(DatasetNoisifier):
    @classmethod
    def preprocess_batch_for_reward_trainer(
        cls,
        examples: dict[str, Any],
    ) -> dict[str, Any]: ...


class ConstructFromConfig(Protocol):
    @classmethod
    def from_config(
        cls,
        cfg: DatasetConfig,
    ) -> datasets.DatasetDict: ...


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
        noise_indices = np.random.choice(
            len(dataset), num_samples_to_add_noise, replace=False
        )

        def maybe_flip_label(example, idx):
            if idx not in noise_indices:
                example["choice"] = abs(example["choice"] - 1)
            return example

        return dataset.map(maybe_flip_label, with_indices=True)

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
    def preprocess_batch_for_reward_trainer(
        cls,
        examples: dict[str, Any],
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
    ) -> dict[str, Any]:
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }

        for summary, choice in zip(examples["summaries"], examples["choice"]):
            chosen = summary[choice]["text"]
            rejected = summary[abs(choice - 1)]["text"]

            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

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
        cfg: DatasetConfig,
    ) -> datasets.DatasetDict:
        dataset_dict = datasets.load_dataset(cls.DATASET_URL, "comparisons")
        assert isinstance(dataset_dict, datasets.DatasetDict)

        processed_train = cls.add_noise(dataset_dict["train"], cfg)
        processed_validation = cls.add_noise(dataset_dict["validation"], cfg)

        if cfg.preprocess_for_reward_trainer:
            assert cfg.preprocess_tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.preprocess_tokenizer, use_fast=True
            )
            processed_train = processed_train.map(
                functools.partial(
                    cls.preprocess_batch_for_reward_trainer,
                    tokenizer=tokenizer,
                ),
                batched=True,
                num_proc=4,
            )
            processed_validation = processed_validation.map(
                functools.partial(
                    cls.preprocess_batch_for_reward_trainer,
                    tokenizer=tokenizer,
                ),
                batched=True,
                num_proc=4,
            )

        return datasets.DatasetDict(
            {
                "train": processed_train,
                "validation": processed_validation,
            }
        )


DATASET_NAME_TO_PREPROCESSOR = {
    "openai/summarize_from_feedback": OpenAIHumanFeedbackDatasetPreprocessor,
}

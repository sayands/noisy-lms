from dataclasses import dataclass
from typing import Any, Protocol, Union

from transformers import AutoTokenizer

@dataclass
class DatasetConfig:
    dataset_path: str = ""
    dataset_noise_type: str = "flip_labels"
    dataset_noise_level: float = 0.0
    dataset_noise_seed: int = 42
    preprocess_for_reward_trainer: bool = False
    preprocess_for_dpo: bool = False
    preprocess_for_ppo: bool = False
    preprocess_for_kto: bool = False
    preprocess_tokenizer: Union[str, None] = None
    preprocess_ppo_tokenizer: AutoTokenizer = None
    preprocess_dpo_tokenizer: AutoTokenizer = None
    preprocess_kto_tokenizer: AutoTokenizer = None
    max_token_length: int = 128

@dataclass
class TokenizerConfig:
    tokenizer_path: str
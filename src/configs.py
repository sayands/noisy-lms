from dataclasses import dataclass
from typing import Any, Protocol, Union

@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_noise_type: str = "flip_labels"
    dataset_noise_level: float = 0.0
    dataset_noise_seed: int = 42
    preprocess_for_reward_trainer: bool = False
    preprocess_tokenizer: Union[str, None] = None
    max_token_length: int = 128
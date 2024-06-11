from dataclasses import dataclass
from typing import Union

import datasets

from .openai_humanfeedback import OpenAIHumanFeedbackDatasetPreprocessor

DATASET_NAME_TO_PREPROCESSOR = {
    "openai/summarize_from_feedback": OpenAIHumanFeedbackDatasetPreprocessor,
}

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
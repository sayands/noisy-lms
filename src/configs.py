from dataclasses import dataclass
from typing import Any, Protocol, Union
import datasets as common

# from src.datasets.openai_humanfeedback import OpenAIHumanFeedbackDatasetPreprocessor

# DATASET_NAME_TO_PREPROCESSOR = {
#     "openai/summarize_from_feedback": OpenAIHumanFeedbackDatasetPreprocessor,
# }

@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_noise_type: str = "flip_labels"
    dataset_noise_level: float = 0.0
    dataset_noise_seed: int = 42
    preprocess_for_reward_trainer: bool = False
    preprocess_tokenizer: Union[str, None] = None

    # def make_dataset(self) -> common.DatasetDict:
    #     return DATASET_NAME_TO_PREPROCESSOR[self.dataset_name].from_config(self)

@dataclass
class TrainingConfig:
    dataset_name: str
    exp_name: str
    output_dir: str
    max_input_length: int = 128
    learning_rate: float = 1e-05
    train_batch_size: int = 16
    eval_batch_size: int = 8
    grad_acc_steps:  int = 1
    max_epoch: int = 5
    eval_steps: int = 1000
    save_steps: int = 1000
    logging_steps: int = 50
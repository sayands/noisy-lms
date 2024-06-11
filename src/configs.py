from dataclasses import dataclass
from typing import Any, Protocol, Union
import datasets as common

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
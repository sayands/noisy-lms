import datasets
from typing import Any, Protocol, Union

from datasets.common import DatasetConfig

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
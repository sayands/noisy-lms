import enum
from typing import Protocol, Union

import torch
from transformers import (  # type: ignore[import]
    AutoModelForSequenceClassification, AutoTokenizer)


class RewardModel(Protocol):
    def get_score(self, context: str, answer: str) -> torch.Tensor: ...


class RewardModelNameHuggingFace(str, enum.Enum):
    DEBERTA_V3_LARGE_V2: str = "OpenAssistant/reward-model-deberta-v3-large-v2"

    def __str__(self) -> str:
        return self.value


class RewardModelHuggingFace(RewardModel):
    def __init__(
        self,
        name: RewardModelNameHuggingFace,
        device: Union[torch.device, str] = "auto",
    ) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(
            self.device
        )

    # TODO(thesstefan): Decide if this needs to be batched
    def get_score(self, context: str, answer: str) -> torch.Tensor:
        inputs = self.tokenizer(context, answer, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        return outputs.logits[0].cpu().detach()


class DeBERTaV3LargeV2(RewardModelHuggingFace):
    def __init__(self, device: Union[torch.device, str] = "auto") -> None:
        super().__init__(
            name=RewardModelNameHuggingFace.DEBERTA_V3_LARGE_V2,
            device=device,
        )

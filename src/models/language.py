import enum
from typing import Protocol, Union

import torch
from transformers import pipeline  # type: ignore[import]


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class SummarizationLanguageModelNameHuggingFace(str, enum.Enum):
    BART_LARGE_CNN: str = "facebook/bart-large-cnn"
    FALCONS_AI_TEXT_SUMMARIZATION: str = "Falconsai/text_summarization"

    def __str__(self) -> str:
        return self.value


class SummarizationLanguageModelHuggingFace(LanguageModel):
    def __init__(self, name: str, device: Union[torch.device, str]) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = pipeline("summarization", model=name, device=self.device)

    def generate(self, prompt: str) -> str:
        return self.model(prompt, max_length=1000, min_length=30, do_sample=False)[0][
            "summary_text"
        ]


class T5FalconSummarization(SummarizationLanguageModelHuggingFace):
    def __init__(self, device: Union[torch.device, str]) -> None:
        super().__init__(
            name=SummarizationLanguageModelNameHuggingFace.FALCONS_AI_TEXT_SUMMARIZATION,
            device=device,
        )


class BartLargeCNNSummarization(SummarizationLanguageModelHuggingFace):
    def __init__(self, device: Union[torch.device, str]) -> None:
        super().__init__(
            name=SummarizationLanguageModelNameHuggingFace.BART_LARGE_CNN,
            device=device,
        )

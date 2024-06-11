import datasets

from .openai_humanfeedback import OpenAIHumanFeedbackDatasetPreprocessor

def make_dataset(dataset_config) -> datasets.DatasetDict:
    if dataset_config.dataset_name == 'openai/summarize_from_feedback':
        return OpenAIHumanFeedbackDatasetPreprocessor.from_config(dataset_config)
    else:
        raise NotImplementedError
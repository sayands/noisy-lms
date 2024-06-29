import datasets

from .openai_humanfeedback import OpenAIHumanFeedbackDatasetPreprocessor

def make_dataset(dataset_config) -> datasets.DatasetDict:
    if dataset_config.dataset_path == 'openai/summarize_from_feedback':
        return OpenAIHumanFeedbackDatasetPreprocessor.from_config(dataset_config)
    else:
        raise NotImplementedError

def build_dataset(trainer_name, dataset_name, tokenizer, **kwargs):
    if trainer_name == 'DPO':
        return OpenAIHumanFeedbackDatasetPreprocessor.dpo_build(dataset_name, tokenizer, 
                                                                kwargs["dataset_noise_level"], kwargs["dataset_noise_seed"])
    if trainer_name == 'nsampler':
        return OpenAIHumanFeedbackDatasetPreprocessor.nsampler_build(dataset_name, tokenizer, kwargs["nsampler_config"])
    if trainer_name == 'PPO':
        return OpenAIHumanFeedbackDatasetPreprocessor.ppo_build(dataset_name, tokenizer)

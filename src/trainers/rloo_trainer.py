import os.path as osp
import os

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl import ModelConfig
from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
from trl.commands.cli_utils import TrlParser
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

import sys

workspace_dir = os.environ["NOISY_LM_DIR"]
src_dir = osp.join(workspace_dir, "src")
sys.path.append(workspace_dir)
sys.path.append(src_dir)


from src.configs import DatasetConfig, TokenizerConfig
from src.datasets_process import make_dataset
from src.utils import common


if __name__ == "__main__":
    parser = TrlParser((RLOOConfig, TokenizerConfig, ModelConfig, DatasetConfig))
    train_config, tokenizer_config, model_config, dataset_config = (
        parser.parse_args_into_dataclasses()
    )

    dataset_noise_level_name = str(dataset_config.dataset_noise_level).split(".")
    dataset_noise_level_name = str(dataset_noise_level_name[0]) + str(
        dataset_noise_level_name[1]
    )

    train_config.exp_name = train_config.run_name + "_" + dataset_noise_level_name # HF API issue
    train_config.output_dir = osp.join(train_config.output_dir, train_config.exp_name)

    common.ensure_dir(train_config.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config.tokenizer_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        train_config.reward_model_path, num_labels=1
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)
    policy = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)

    dataset_config.preprocess_rloo_tokenizer = tokenizer
    dataset = make_dataset(dataset_config)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    dataset_text_field = "prompt"

    trainer = RLOOTrainer(
        config=train_config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(train_config.output_dir)
    trainer.generate_completions()

"""
python $NOISY_LM_DIR/src/trainers/ppo_trainer.py \
    --output_dir /home/anghosh/PROJECT/output/ \
    --save_steps 100 \
    --exp_name train_ppo \
    --task_name text-classification \
    --reward_model /home/anghosh/PROJECT/train_rm_00/checkpoint-29000 \
    --model_name /home/anghosh/PROJECT/sft_gpt2/checkpoint-36000 \
    --tokenizer_path gpt2 \
    --max_length 128 \
    --dataset_name openai/summarize_from_feedback \
    --dataset_noise_type flip_labels \
    --dataset_noise_level 0.01 \
    --dataset_noise_seed 42 \
    --preprocess_for_ppo \
    --learning_rate 1e-03 \
    --batch_size 4 \
    --mini_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --ppo_epochs 5 \
    --steps 1000 \
    --log_with wandb \
"""

import sys
import os
import os.path as osp
workspace_dir = os.environ['NOISY_LM_DIR']
src_dir = osp.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)

import torch
from src.datasets_process import build_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from src.configs import DatasetConfig, TokenizerConfig, PPOSaveConfig

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.commands.cli_utils import TrlParser
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from src.utils import common

tqdm.pandas()
device = 0 if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    #Parsing
    parser = TrlParser((PPOSaveConfig, PPOConfig, DatasetConfig, TokenizerConfig))
    args, ppo_config, dataset_config, tokenizer_config = parser.parse_args_into_dataclasses()

    #Name and Output Directory Creation
    dataset_noise_level_name = str(dataset_config.dataset_noise_level).split('.')
    dataset_noise_level_name = str(dataset_noise_level_name[0]) + str(dataset_noise_level_name[1])
    ppo_config.exp_name =  ppo_config.exp_name + '_' + dataset_noise_level_name
    args.output_dir = osp.join(args.output_dir, ppo_config.exp_name)
    common.ensure_dir(args.output_dir)

    # set seed before initializing value head for deterministic eval
    set_seed(dataset_config.dataset_noise_seed)

    #Build the model, the reference model, and the tokenizer.
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name, trust_remote_code=True)
    device_map = None
    peft_config = None

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        trust_remote_code=True,
        device_map=device_map,
        peft_config=peft_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    
    #Dataset loading
    dataset = build_dataset('PPO', dataset_config.dataset_path, tokenizer)
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["validation"]
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # Build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    # We then build the reward model pipeline
    sent_kwargs = {"top_k":None, "function_to_apply": "none", "batch_size": ppo_config.batch_size}
    task, reward_model_name = ppo_config.task_name, ppo_config.reward_model
    reward_pipe = pipeline(task, model=reward_model_name, device=device)
    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = tokenizer.pad_token_id

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 32,
    }

    for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute reward sentiment
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_pipe_outputs = reward_pipe(ref_texts, **sent_kwargs)
        ref_rewards = [torch.tensor(output[0]["score"]) for output in ref_pipe_outputs]
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
        print("epoch", _epoch)

    #save model
    ppo_trainer.save_pretrained(args.output_dir)
import torch
import typing
from typing import Callable, List, Optional, Union

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from accelerate.utils import gather_object
import wandb
from src.utils import common

def log_ppo_stats(trainer: PPOTrainer,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
        columns_to_log: typing.Iterable[str] = ("query", "response")):
    
    if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards).to(trainer.current_device)
    rewards = trainer.accelerator.gather(rewards).flatten()
    batch_list = [batch[column_to_log] for column_to_log in columns_to_log]

    if trainer.is_distributed:
        gathered_batch_list = []
        for b in batch_list:
            flattened = gather_object(b)
            gathered_batch_list.append(flattened)
        batch_list = gathered_batch_list
    
    logs = {}

    # Log stats
    if "query" not in batch.keys() and "response" not in batch.keys():
        # warn the user that the game logs will not be logged
        warnings.warn(
            "The game logs will not be logged because the batch does not contain the keys 'query' and "
            "'response'. "
        )
    
    table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
    logs.update({"game_log": wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)})

    logs.update(stats)

    # manually cast in fp32 for bf16 torch tensors
    for k, v in logs.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
            logs[k] = v.float()

    logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
    logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
    logs["env/reward_dist"] = rewards.cpu().numpy()
    
    trainer.accelerator.log(
                logs,
                step=trainer.current_step)
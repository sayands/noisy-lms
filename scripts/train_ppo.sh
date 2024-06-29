export NOISY_LM_DIR=$(pwd)
export WANDB_PROJECT="noisy-lms"
export WANDB_LOG_MODEL="checkpoint"

python $NOISY_LM_DIR/src/trainers/ppo_trainer.py \
    --output_dir /home/anghosh/PROJECT/output/ \
    --save_steps 100 \
    --exp_name train_ppo \
    --task_name text-classification \
    --reward_model /home/anghosh/PROJECT/train_rm_00/checkpoint-29000 \
    --model_name /home/anghosh/PROJECT/sft_gpt2/checkpoint-36000 \
    --tokenizer_path gpt2 \
    --max_token_length 128 \
    --dataset_path openai/summarize_from_feedback \
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
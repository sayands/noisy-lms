export NOISY_LM_DIR=$(pwd)
export WANDB_PROJECT="noisy-lms"
export WANDB_LOG_MODEL="checkpoint"

python $NOISY_LM_DIR/src/trainers/dpo_trainer_2.py \
    --dataset_name openai/summarize_from_feedback \
    --model_name_or_path gpt2 \
    --dataset_noise_level 0.05 \
    --dataset_noise_seed 42 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 50 \
    --eval_strategy steps \
    --eval_steps 500 \
    --warmup_steps 150 \
    --report_to wandb \
    --logging_first_step \
    --no_remove_unused_columns \
    --output_dir /home/anghosh/PROJECT/output \
    --run_name dpo_trainer_wnoise \

export NOISY_LM_DIR=$(pwd)
export WANDB_PROJECT="noisy-lms"
export WANDB_LOG_MODEL="checkpoint"

python $NOISY_LM_DIR/src/trainers/rloo_trainer.py \
    --output_dir out \
    --run_name train_rloo \
    --reward_model_path gpt2 \
    --model_name_or_path gpt2 \
    --tokenizer_path gpt2 \
    --dataset_name openai/summarize_from_feedback \
    --dataset_noise_type flip_labels \
    --dataset_noise_level 0.01 \
    --dataset_noise_seed 42 \
    --preprocess_for_rloo \
    --total_episodes 10000 \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 50 
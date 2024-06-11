export NOISY_LM_DIR=$(pwd)
export WANDB_PROJECT="noisy-lms"
export WANDB_LOG_MODEL="checkpoint"

python $NOISY_LM_DIR/src/trainers/reward_model_trainer.py \
    --output_dir /media/sdebsarkar/extra-hdd/CSNLP \
    --run_name train_rm_gpt2 \
    --model_name_or_path gpt2 \
    --max_length 128 \
    --dataset_name openai/summarize_from_feedback \
    --dataset_noise_type flip_labels \
    --dataset_noise_level 0.0 \
    --dataset_noise_seed 42 \
    --preprocess_for_reward_trainer \
    --preprocess_tokenizer gpt2 \
    --learning_rate 1e-05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_steps 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 50 
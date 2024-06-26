export WANDB_PROJECT="noisy-lms"
export NOISY_LM_DIR=/home/users/sdsarkar/code/CSNLP/noisy-lms

cd $NOISY_LM_DIR
python $NOISY_LM_DIR/src/trainers/rloo_trainer.py \
    --output_dir /scratch/users/sdsarkar/CSNLP \
    --report_to wandb \
    --run_name train_rloo \
    --reward_model_path /scratch/users/sdsarkar/CSNLP/train_rm_00/checkpoint-29000 \
    --model_name_or_path /scratch/users/sdsarkar/CSNLP/train_sft_gpt2/checkpoint-36000 \
    --tokenizer_path gpt2 \
    --dataset_path openai/summarize_from_feedback \
    --dataset_noise_type flip_labels \
    --dataset_noise_level 0.0 \
    --dataset_noise_seed 42 \
    --preprocess_for_rloo \
    --total_episodes 10000 \
    --learning_rate 1e-3 \
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
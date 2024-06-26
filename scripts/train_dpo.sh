export WANDB_PROJECT="noisy-lms"
export NOISY_LM_DIR=/home/users/sdsarkar/code/CSNLP/noisy-lms

export HF_HOME=/scratch/users/sdsarkar/hf_cache/
export HF_DATASETS_CACHE=/scratch/users/sdsarkar/hf_cache/datasets

cd $NOISY_LM_DIR
python $NOISY_LM_DIR/src/trainers/dpo_trainer.py \
    --dataset_path openai/summarize_from_feedback \
    --model_name_or_path /scratch/users/sdsarkar/CSNLP/train_sft_gpt2/checkpoint-36000 \
    --tokenizer_path gpt2 \
    --dataset_noise_level 0.2 \
    --dataset_noise_seed 42 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 50 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --save_steps 1000 \
    --warmup_steps 150 \
    --report_to wandb \
    --logging_first_step \
    --no_remove_unused_columns \
    --output_dir /scratch/users/sdsarkar/CSNLP \
    --run_name train_dpo


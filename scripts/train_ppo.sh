export WANDB_PROJECT="noisy-lms"
export NOISY_LM_DIR=/home/users/sdsarkar/code/CSNLP/noisy-lms

export HF_HOME=/scratch/users/sdsarkar/hf_cache/
export HF_DATASETS_CACHE=/scratch/users/sdsarkar/hf_cache/datasets

cd $NOISY_LM_DIR
python $NOISY_LM_DIR/src/trainers/ppo_trainer.py \
    --output_dir /scratch/users/sdsarkar/CSNLP \
    --save_steps 100 \
    --tracker_project_name noisy-lms \
    --exp_name train_ppo \
    --task_name text-classification \
    --reward_model /scratch/users/sdsarkar/CSNLP/train_rm_00/checkpoint-29000 \
    --model_name /scratch/users/sdsarkar/CSNLP/train_sft_gpt2/checkpoint-36000 \
    --tokenizer_path gpt2 \
    --max_token_length 128 \
    --dataset_path openai/summarize_from_feedback \
    --dataset_noise_type flip_labels \
    --dataset_noise_level 0.0 \
    --dataset_noise_seed 42 \
    --preprocess_for_ppo \
    --learning_rate 1e-3 \
    --batch_size 4 \
    --mini_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --ppo_epochs 5 \
    --steps 1000 \
    --log_with wandb \

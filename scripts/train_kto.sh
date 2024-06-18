export NOISY_LM_DIR=$(pwd)
export WANDB_PROJECT="noisy-lms"

python $NOISY_LM_DIR/src/trainers/kto_trainer.py \
    --output_dir /scratch/users/sdsarkar/CSNLP/ \
    --run_name train_kto \
    --model_name_or_path /scratch/users/sdsarkar/CSNLP/train_sft_gpt2/checkpoint-36000 \
    --tokenizer_path /scratch/users/sdsarkar/CSNLP/rm_gpt/train_rm_gpt2_00 \
    --max_length 128 \
    --dataset_name openai/summarize_from_feedback \
    --dataset_noise_type flip_labels \
    --dataset_noise_level 0.0 \
    --dataset_noise_seed 42 \
    --preprocess_for_kto \
    --learning_rate 1e-3 \
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

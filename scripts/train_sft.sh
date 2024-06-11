export NOISY_LM_DIR=$(pwd)
export WANDB_PROJECT="noisy-lms"
export WANDB_LOG_MODEL="checkpoint"

python $NOISY_LM_DIR/src/trainers/supervised_finetune_trainer.py \
        --dataset_name "CarperAI/openai_summarize_tldr" \
        --output_dir /media/sdebsarkar/extra-hdd/CSNLP \
        --run_name train_sft_gpt2 \
        --max_length 128 \
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


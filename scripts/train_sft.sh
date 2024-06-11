export NOISY_LM_DIR=$(pwd)
export WANDB_PROJECT="noisy-lms"
export WANDB_LOG_MODEL="checkpoint"

python $NOISY_LM_DIR/src/trainers/supervised_finetune_trainer.py --dataset_name "CarperAI/openai_summarize_tldr" \
        --output_dir /media/sdebsarkar/extra-hdd/CSNLP \
        --exp_name train_sft_gpt2
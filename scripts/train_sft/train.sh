export NOISY_LM_DIR=$(pwd)
export WANDB_PROJECT="noisy-lms"
export WANDB_LOG_MODEL="checkpoint"

cd $NOISY_LM_DIR
python src/trainers/supervised_finetune_trainer.py --config $NOISY_LM_DIR/scripts/train_sft/train_sft.yaml
export NOISY_LM_DIR=$(pwd)
cd $NOISY_LM_DIR

python $NOISY_LM_DIR/src/metrics/generate_kl.py \
    --lm_reward_score_csv_path /home/anghosh/PROJECT/output/eval_rewards.csv \
    --ref_model_reward_score_csv_path /home/anghosh/PROJECT/output/eval_rewards.csv \

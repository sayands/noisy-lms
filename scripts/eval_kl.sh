export NOISY_LM_DIR=/local/home/sdebsarkar/Documents/code/CSNLP-Project/noisy-lms
export HF_HOME=/media/sdebsarkar/extra-hdd/CSNLP/hf_cache
export HF_DATASETS_CACHE=/media/sdebsarkar/extra-hdd/CSNLP/hf_cache/datasets

cd $NOISY_LM_DIR
python $NOISY_LM_DIR/src/metrics/generate_kl.py \
    --lm_reward_score_csv_path /media/sdebsarkar/extra-hdd/CSNLP/reward_output/eval_dpo_02_val_temp.csv \
    --ref_model_reward_score_csv_path /media/sdebsarkar/extra-hdd/CSNLP/reward_output/ref.csv

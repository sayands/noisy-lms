export NOISY_LM_DIR=/local/home/sdebsarkar/Documents/code/CSNLP-Project/noisy-lms
export HF_HOME=/media/sdebsarkar/extra-hdd/CSNLP/hf_cache
export HF_DATASETS_CACHE=/media/sdebsarkar/extra-hdd/CSNLP/hf_cache/datasets

cd $NOISY_LM_DIR
python $NOISY_LM_DIR/src/metrics/generate_reward_scores.py \
    --dataset_path openai/summarize_from_feedback \
    --lm_model_name_or_path /media/sdebsarkar/extra-hdd/CSNLP/train_rloo_02 \
    --reward_model_name_or_path /media/sdebsarkar/extra-hdd/CSNLP/train_rm_00/checkpoint-29000 \
    --tokenizer_path gpt2 \
    --sampling_seed 42 \
    --sample_input_length \
    --input_min_text_length 50 \
    --input_max_text_length 128 \
    --output_min_length 10 \
    --output_max_length 25 \
    --n_best_of 4 \
    --num_samples 500 \
    --num_beams 1 \
    --no_repeat_ngram_size 0 \
    --temperature 0.25 \
    --top_k 0 \
    --top_p 1.0 \
    --do_sample \
    --csv_save_dir /media/sdebsarkar/extra-hdd/CSNLP/reward_output/eval_rloo_02_val_temp.csv
#additional inputs: default set to false, need to add for other kind of sampling generations
# --early_stopping \
# --do_sample \

# ---------------------------------------------------------------------------
# Experiments
# PPO train & val: 0.0, 0.01, 0.05, 0.1, 0.2
# DPO train & val: 0.0, 0.01, 0.05, 0.1, 0.2
# RLOO train & val: 0.0, 0.01, 0.05, 0.1, 0.2
# RM train & val: 0.0, 0.01, 0.05, 0.1, 0.2

# Greedy:
#   default: PPO train + DPO train + RLOO train
#   default: PPO val + DPO val + RLOO val
#   nsampling: RM train (model: SFT, n_best_of: 8)
#   nsampling: RM val (model: SFT, n_best_of: 8)

# Temperature Sampling (temperature: 0.25, do_sample):
#   default: PPO val + DPO val + RLOO val
#   nsampling: RM val (model: SFT, n_best_of: 8)
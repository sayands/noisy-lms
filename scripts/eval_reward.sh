export NOISY_LM_DIR=$(pwd)
cd $NOISY_LM_DIR

python $NOISY_LM_DIR/src/metrics/generate_reward_scores.py \
    --dataset_path openai/summarize_from_feedback \
    --lm_model_name_or_path /home/anghosh/PROJECT/sft_gpt2/checkpoint-36000 \
    --reward_model_name_or_path /home/anghosh/PROJECT/train_rm_00/checkpoint-29000 \
    --tokenizer_path gpt2 \
    --sampling_seed 42 \
    --sample_input_length \
    --input_min_text_length 50 \
    --input_max_text_length 128 \
    --output_min_length 10 \
    --output_max_length 25 \
    --n_best_of 1 \
    --num_samples 2000 \
    --num_beams 1 \
    --no_repeat_ngram_size 0 \
    --temperature 1.0 \
    --top_k 0 \
    --top_p 1.0 \
    --csv_save_dir /home/anghosh/PROJECT/output/eval_rewards.csv \

#additional inputs: default set to false, need to add for other kind of sampling generations
    # --early_stopping \
    # --do_sample \
export NOISY_LM_DIR=$(pwd)
cd $NOISY_LM_DIR

python $NOISY_LM_DIR/src/trainers/n_sampler.py \
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
    --n_best_of 4 \
    --num_samples 10 \
    --csv_save_dir /home/anghosh/PROJECT/output/nsampler.csv \
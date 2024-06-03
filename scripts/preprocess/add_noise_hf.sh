export NOISY_LM_DIR='/local/home/sdebsarkar/Documents/code/CSNLP-Project/noisy_lms'

cd $NOISY_LM_DIR
python preprocessing/add_noise.py --config $NOISY_LM_DIR/scripts/preprocess/add_noise.yaml
export NOISY_LM_DIR=$(pwd)

cd $NOISY_LM_DIR
python preprocessing/add_noise.py --config $NOISY_LM_DIR/scripts/preprocess/add_noise.yaml --split train
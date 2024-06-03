export NOISY_LM_DIR=$(pwd)

cd $NOISY_LM_DIR
python download.py --config $NOISY_LM_DIR/scripts/download/download.yaml
### Dependencies :memo:
The main dependencies of the project are the following:
```yaml
python: 3.9.12
cuda: 11.7
```

### Installation
Set up a python venv as follows :
```bash
git clone git@github.com:sayands/noisy-lms.git
cd noisy_lms
python3.9 -m venv <venvs_dir>/noisenv
source <venvs_dir>/noisenv/bin/activate
pip install -r requirements.txt
```

### Usage
OpenAI Human Feedback Data download using [Huggingface](https://huggingface.co/datasets/openai/summarize_from_feedback):

- Change `NOISY_LM_DIR` in `scripts/download/download_hf.sh` and `data:root_dir` in `scripts/download/download.yaml`
```bash
bash scripts/download/download_hf.sh
```

Flip labels using:
- Change `NOISY_LM_DIR` in `scripts/preprocess/add_noise_hf.sh` and `data:root_dir` in `scripts/preprocess/add_noise.yaml`
- Change `preprocess:noise_level` in `scripts/preprocess/add_noise.yaml` (Remember noise level is in percentage of the no.of samples in `train` data split!)
```bash
bash scripts/preprocess/add_noise_hf.sh
```



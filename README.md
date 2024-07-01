<div align='center'>
  <h2 align="center">How Much Noise is Too Much Noise? </h2>
  <h3 align="center">Ankita Ghosh, Sayan Deb Sarkar, Stefan Stefanache </h3>
  Computational Semantics For Natural Language Processing - Spring 2024, ETH Zurich 
  
  <h3 align="center"> Professor: <a href="https://www.mrinmaya.io/">Mrinmaya Sachan</a> </h3>
  <h3 align="center"> Project Supervisor: <a href="https://shehzaadzd.github.io/">Shehzaad Dhuliawala</a> </h3>

</div>

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

### Dataset
We use the [OpenAI Summarize From Human Feedback Dataset](https://huggingface.co/datasets/openai/summarize_from_feedback) and add noise to the dataset on-the-go during our experiments.
> **_NOTE:_** Fixing the random seed in our experiments makes for reproducible results.

### Usage - SFT + Reward Models
Please change the `HF_HOME` and `output_dir` in the script files before running. All experiment runs would automatically be logged into `wandb` into the project name desired.
> **_NOTE:_**  By default, we use a `gpt2` model from HF hub, it can be changed by the corresponding parameter in the scripts.

#### Train Supervised Fine-Tuned Model 

```bash 
bash scripts/train_sft.sh
```

#### Train Reward Models
Change the `dataset_noise_level` parameter according to needs. For our experiments, we train on noise levels `0.0, 0.01, 0.05, 0.1, 0.2`. Run

```bash
bash scripts/train_reward_model.sh
```

### Usage - Reinforcement Learning Pipelines
Change, `dataset_noise_levels`, `model_name` and `reward_model_name` parameters according to the dataset needs.

#### Train Proximal Policy Optimization (PPO)

```bash
bash scripts/train_ppo.sh
```

#### Train Direct Preference Optimization (DPO)

```bash
bash scripts/train_dpo.sh
```

#### Train REINFORCE Leave-One-Out (RLOO)

```bash
bash scripts/train_rloo.sh
```

### Evaluation - Reward Scores & KL Divergence

#### Reward Score
To generate results for Reward Scores of N-Sampling, PPO, RLOO, and DPO on training and validation prompts, change the corresponding params and run:

```bash
bash scripts/eval_reward.sh
```

#### KL Divergence
Change the parameters of the files in the script and run:
```bash
bash scripts/eval_kl.sh
```

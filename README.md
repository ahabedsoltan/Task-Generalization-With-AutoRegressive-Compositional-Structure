# Task Generalization With AutoRegressive Compositional Structure

This repository contains the code to reproduce the results from our paper [Task Generalization With AutoRegressive Compositional Structure: Can Learning From $D$ Tasks Generalize to $D^{T}$ Tasks?](https://arxiv.org/abs/2502.08991) We investigate task generalization through the lens of AutoRegressive Compositional (ARC) structure, where each task is a composition of $T$ operations, and each operation is among a finite family of $D$ subtasks. We show that generalization to $D^T$ tasks is theoretically achievable by training on only $\tilde O (D) $ tasks. Empirically, we demonstrate that Transformers achieve exponential task generalization on sparse parity, arithmetic and language translation tasks. 

<img src="https://files.catbox.moe/44udut.png" alt="alt text" width="70%" />

### Setup

#### Environment
Please install the conda environment by running 
```
conda env create -f environment.yaml
```
#### Wandb
### Weights & Biases (wandb) Setup

This code **requires** `wandb` to run. Follow the steps below to set up `wandb`:

#### 1. Install `wandb`:
```bash
pip install wandb
```

#### 2. Login to `wandb`:
```bash
wandb login
```
Or, if you're using API keys directly:
```bash
export WANDB_API_KEY=your_api_key_here
```

#### 3. Configure the following parameters in 'main.py' script before running:
```python
wandb_init = {
    "project_name": "<your_project_name>",
    "mode": "online",  # or 'offline'
    "key": "<your_wandb_api_key>",
    "org": "<your_wandb_entity>"
}

os.environ["WANDB_API_KEY"] = wandb_init['key']
os.environ["WANDB_MODE"] = wandb_init['mode']
run = wandb.init(project=wandb_init['project_name'], entity=wandb_init['org'])
```

Make sure to replace `<your_project_name>`, `<your_wandb_api_key>`, and `<your_wandb_entity>` with your actual `wandb` project settings.


### Running the Experiments

![alt text](https://files.catbox.moe/mqcqa9.png)

#### Sparse Parity Task
Use the following command to run the experiment:


Standard Training (Section 4)
```bash
python main.py --dim_ambient 15 --dim_k 3 --context_length 40 --n_layers 3 --n_heads 1 --model_type gpt2 \
               --max_lr 8e-5 --gpt2_width 192 --n_train_tasks 121 --cot 1 --n_train_total 25000 --sort_tuples 1
```
Training with Missing Coordinate (Section 5.1)
```bash
python main.py --dim_ambient 10 --dim_k 3 --context_length 40 --n_layers 3 --n_heads 1 --model_type gpt2 \
               --max_lr 8e-5 --gpt2_width 192 --n_train_tasks 121 --cot 1 --n_train_total 25000 --sort_tuples 1 ----missing_coordinate 2 5
```

Training with Missing Pair (Section 5.1)
```bash
python main.py --dim_ambient 10 --dim_k 3 --context_length 40 --n_layers 3 --n_heads 1 --model_type gpt2 \
               --max_lr 8e-5 --gpt2_width 192 --n_train_tasks 121 --cot 1 --n_train_total 25000 --sort_tuples 1 ----missing_pair 4 6
```

### Argument Descriptions:

- `--sort_tuples`: Whether to sort the tuples (default: `1`). Set to `0` to **ignore order**, allowing secret keys like `(3,1,2)` instead of always sorted `(1,2,3)`.
- `--dim_k`: Number of secret keys, corresponds to **k** in the paper.
- `--dim_ambient`: Ambient dimension, corresponds to **d** in the paper.
- `--n_train_total`: Total number of training samples, **evenly divided** among tasks.
- `--n_train_tasks`: Number of training tasks, randomly selected from all possible tasks (i.e., from the total of $\binom{d}{k}$ or $d!/(n-k)!$ tuples for sorted or unsorted cases respectively).
- `--missing_coordinate c v`:  Tasks where the $c$-th coordinate takes the value $v$ are excluded from training set. 
- `--missing_pairs (v1, v2)`: Tasks that contain both $v_1$ and $v_2$ in the tuples are excluded from training set.

### Citation

Please cite us:

```
@article{abedsoltan2025task,
  title={Task Generalization With AutoRegressive Compositional Structure: Can Learning From $D$ Tasks Generalize to $D^T$ Tasks?},
  author={Abedsoltan, Amirhesam and Zhang, Huaqing and Wen, Kaiyue and Lin, Hongzhou and Zhang, Jingzhao and Belkin, Mikhail},
  journal={arXiv preprint arXiv:2502.08991},
  year={2025}
}
```
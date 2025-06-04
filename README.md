# Task Generalization With AutoRegressive Compositional Structure

This repository contains the code to reproduce the results from our paper [Task Generalization With AutoRegressive Compositional Structure: Can Learning From D Tasks Generalize to D^T Tasks?](https://arxiv.org/abs/2502.08991) We investigate task generalization through the lens of AutoRegressive Compositional (ARC) structure, where each task is a composition of $T$ operations, and each operation is among a finite family of $D$ subtasks. We theoretically demonstrate that generalization to $D^T$ tasks can be achieved by training on only a nearly linear number of tasks in $D$. Empirically, we demonstrate that Transformers achieve exponential task generalization on sparse parity, arithmetic and language translation tasks. 




## Paper Overview

### Task Generalization: Beyond Conventional Learning


<img src="https://files.catbox.moe/53qjef.png" alt="alt text" width="70%" />

- Traditional Learning: Approximate a target function $f^\star \in  F$ using training examples and generalize to new inputs.

- Our Focus: **Task Generalization**. *Can a model trained on a subset of tasks* $F_{train}$ *generalize to all tasks in* $F$*, including unseen ones?*

  - We show that this is theoretically achievable: when the task class ${F}$ admits an **AutoRegressive Compositional (ARC)** structure, generalization to exponentially many unseen tasks can be achieved by training on only a nearly linear number of tasks.

  - *AutoRegressive Compositional Task Class*: Each task in the class consists of $T$ subtasks, each defined by a parameter $\theta_1,\cdots,\theta_T$ respectively. Every subtask offers $D$ options, resulting in a total of $D^T$ composed tasks. Tokens are generated in an autoregressive manner, as illustrated in the figure below.
    <img src="https://files.catbox.moe/c4rlpb.png" alt="alt text" width="70%" />
  
### Empirical Experiments: Sparse Parity Case Study

####  Synthetic task: sparse parity problem. 

- Sparse Parity: Compute the XOR of $k$ out of $d$ indices of a binary sequence.
- Example: 

  - $x = (\mathbf {1}, \mathbf 0, 1, \mathbf 0, 0)$ &nbsp;&nbsp;&nbsp;  (binary sequence of size $d$)

  - $S = (1,2,4)$ &nbsp;&nbsp;&nbsp; (secret keys of size $k$ -- corresponding to a task $f\in  F$)

  - Output: $x[1] \oplus x [2] \oplus x[4] = \mathbf 1 \oplus \mathbf 0 \oplus \mathbf 0  = 1 $.

#### Learning Sparse Parity with in-context learning (ICL)

- Transformers with ICL struggle with learning sparse parity.

- In contrast, incorporating Chain-of-Thought (CoT) reasoning allows Transformers to easily generalize to unseen tasks by introducing AutoRegressive Structure. 

<p float="left">
  <img src="https://files.catbox.moe/wany7o.png" width="375"/>
  <img src="https://files.catbox.moe/fa178z.png" width="300"/>
</p>

| $d$ | $k$ | # Training Tasks | # Total Tasks | Accuracy (%) |
|-----|-----|------------------|----------------|---------------|
| 10  |  5  | 69               | 252            | 98.51         |
| 15  |  7  | 121              | 6400           | 99.12         |
| 20  | 10  | 180              | 185000         | 98.67         |
| 25  | 12  | 241              | 3200000        | 98.60         |
| 30  | 15  | 306              | 155000000      | 98.10         |

*Table: Task generalization performance as d and k increase. Traning on only a nearly linear number of tasks enables generalization to exponentially many unseen tasks in the parity function family.*


For other results and details, check out our paper [https://arxiv.org/abs/2502.08991](https://arxiv.org/abs/2502.08991)! 

## Setup

### Environment
Please install the conda environment by running 
```
conda env create -f environment.yaml
```
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


## Running the Experiments

### Sparse Parity Task

<img src="https://files.catbox.moe/94yjz5.png" alt="alt text" width="70%" />

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

#### Argument Descriptions:

- `--sort_tuples`: Whether to sort the tuples (default: `1`). Set to `0` to **ignore order**, allowing secret keys like `(3,1,2)` instead of always sorted `(1,2,3)`.
- `--dim_k`: Number of secret keys, corresponds to **k** in the paper.
- `--dim_ambient`: Ambient dimension, corresponds to **d** in the paper.
- `--n_train_total`: Total number of training samples, **evenly divided** among tasks.
- `--n_train_tasks`: Number of training tasks, randomly selected from all possible tasks (i.e., from the total of $\binom{d}{k}$ or $d!/(n-k)!$ tuples for sorted or unsorted cases respectively).
- `--missing_coordinate c v`:  Tasks where the $c$-th coordinate takes the value $v$ are excluded from training set. 
- `--missing_pairs (v1, v2)`: Tasks that contain both $v_1$ and $v_2$ in the tuples are excluded from training set.

### Multi-Step Language Translation Task

<img src="https://files.catbox.moe/am1vcw.png" alt="alt text" width="70%" />

First, navigate to the project directory:
```bash
cd TranslationComposition
```

To generate training data, run:
```bash
python src/generate_data_composition.py --config_path config/compose_data_config.py
``` 
Customize data generation with the following optional arguments in `config/compose_data_config.py`:

- `n_samples`: Number of training samples, default is 100,000. 
- `n_lines`:  Number of in-context examples (each in-context example contains tokens in different languages)
- `n_steps`: Number of languages in each in-context example.
- `n_tasks`: Number of language combinations in training dataset.
- `num_languages`: Total number of languages to include in the dataset.

Once the data is generated, start training by running:
```bash
bash scripts/train.sh
```

## Citation

Please cite us:

```
@article{abedsoltan2025task,
  title={Task Generalization With AutoRegressive Compositional Structure: Can Learning From $D$ Tasks Generalize to $D^T$ Tasks?},
  author={Abedsoltan, Amirhesam and Zhang, Huaqing and Wen, Kaiyue and Lin, Hongzhou and Zhang, Jingzhao and Belkin, Mikhail},
  journal={arXiv preprint arXiv:2502.08991},
  year={2025}
}
```
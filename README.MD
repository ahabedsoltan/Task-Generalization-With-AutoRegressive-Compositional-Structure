### Running the Experiment

Use the following command to run the experiment:

```bash
python main.py --dim_ambient 15 --dim_k 3 --context_length 40 --n_layers 3 --n_heads 1 --model_type gpt2 \
               --max_lr 8e-5 --gpt2_width 192 --n_train_tasks 121 --cot 1 --n_train_total 25000 --sort_tuples 1
```

### Argument Descriptions:

- `--sort_tuples`: Whether to sort the tuples (default: `1`). Set to `0` to **ignore order**, allowing secret keys like `(3,1,2)` instead of always sorted `(1,2,3)`.
- `--dim_k`: Number of secret keys, corresponds to **k** in the paper.
- `--dim_ambient`: Ambient dimension, corresponds to **d** in the paper.
- `--n_train_total`: Total number of training samples, **evenly divided** among tasks.
- `--n_train_tasks`: Number of training tasks, randomly selected from all possible tasks (i.e., from the total of $\binom{d}{k}$ ).

---

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




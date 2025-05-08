### Running the Experiment

Use the following command to run the experiment:

```bash
python main.py --dim_ambient 15 --dim_k 3 --context_length 40 --n_layers 3 --n_heads 1 --model_type gpt2 \
               --max_lr 8e-5 --gpt2_width 192 --n_train_tasks 121 --cot 1 --n_train_total 25000 --sort_tuples 1
```

#### Argument Descriptions:

- `--sort_tuples`: Whether to sort the tuples (default: `1`). Set to `0` to **ignore order**, allowing secret keys like `(3,1,2)` instead of always sorted `(1,2,3)`.
- `--dim_k`: Number of secret keys, corresponds to **k** in the paper.
- `--dim_ambient`: Ambient dimension, corresponds to **d** in the paper.
- `--n_train_total`: Total number of training samples, **evenly divided** among tasks.
- `--n_train_tasks`: Number of training tasks, randomly selected from all possible tasks (i.e., from the total of "d choose k").

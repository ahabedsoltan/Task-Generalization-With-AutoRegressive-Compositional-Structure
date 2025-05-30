import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, os, random,  wandb, argparse, itertools, random, math

from data import create_dataset
from utils import set_all_seeds, evaluate_model
from itertools import combinations, permutations
from datetime import datetime
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW,SGD
from torch.utils.data import Dataset, DataLoader






# Initialize the argument parser
parser = argparse.ArgumentParser(description="Description of your script.")
# Add arguments
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--n_heads', type=int, default=3)
parser.add_argument('--context_length', type=int, default=1)
parser.add_argument('--data_method', type=str, default="n_train_task")
parser.add_argument('--model_type', type=str, default="gpt")
parser.add_argument('--pre_trained', type=int, default=0)
parser.add_argument('--dim_ambient', type=int, default=10)
parser.add_argument('--dim_k', type=int, default=3)
parser.add_argument('--split_type', type=int, default=1)
parser.add_argument('--gpt2_width', type=int, default=192)
parser.add_argument('--max_lr', type=float, default=6e-5)
parser.add_argument('--train_task_perc', type=float, default=0.8)
parser.add_argument('--n_train_tasks', type=int, default=1)
parser.add_argument('--n_train_seqs_exponent', type=int, default=9)
parser.add_argument('--n_test_seqs_exponent', type=int, default=9)
parser.add_argument('--reg_position_embeddings', type=int, default=0)
parser.add_argument('--n_train_total', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--sort_tuples', type=int, default=1)
parser.add_argument('--cot', type=int, default=0)
parser.add_argument('--select_sequences', type=int, default=0)
parser.add_argument('--model_save_path',type=str,default='./checkpoints',help='Path to directory where the model will be saved')
parser.add_argument('--missing_pair', type=int, nargs='+', default=None, help='List of missing pair')
parser.add_argument('--missing_coordinate', type=int, nargs='+', default=None, help='missing secret coordinate')





# Parse arguments
args = parser.parse_args()


#wandb
wandb_init = {}
wandb_init["project_name"] = ""
wandb_init["mode"] = ''
wandb_init["key"] = ""
wandb_init["org"] = ""

os.environ["WANDB_API_KEY"] = wandb_init['key']
os.environ["WANDB_MODE"] = wandb_init['mode']  # online or offline
run = wandb.init(project=wandb_init['project_name'], entity=wandb_init['org'])



# Example usage
SEED = args.seed
set_all_seeds(SEED)
d = args.dim_ambient
k = args.dim_k
#ICL data non overlaping
N = args.context_length
vocab_size = 2


np.random.seed(SEED)
random.seed(SEED)
if args.sort_tuples == 1:
    all_tuples = list(combinations(range( d ), k))
else:

    all_tuples = list(permutations(range(d), k))



if args.data_method == "dlogd":
    if args.missing_pair != None:
        train_tuples = [t for t in all_tuples if (args.missing_pair[0] not in t or args.missing_pair[1] not in t)]
    elif args.missing_coordinate != None:
        train_tuples = [t for t in all_tuples if t[args.missing_coordinate[0]-1] !=  args.missing_coordinate[1]]
    else:
        train_tuples = all_tuples
    train_set = sorted(list(set(random.sample( train_tuples, int( 2*d*math.log(d) ) ) ))) #+ [tuple(range(i, i + k)) for i in range( d - k + 1)]
    test_set = [t for t in all_tuples if t not in train_set]
    test_set = random.sample(test_set, min(len(test_set), 200))

else:
    test_set = random.sample(all_tuples,min(200,len(all_tuples) -args.n_train_tasks ) )
    train_tuples = []
    if args.missing_pair != None:
        train_tuples = [t for t in all_tuples if (args.missing_pair[0] not in t or args.missing_pair[1] not in t) and t not in test_set]
    elif args.missing_coordinate != None:
        train_tuples = [t for t in all_tuples if t[args.missing_coordinate[0]-1] !=  args.missing_coordinate[1] and t not in test_set]
    else:
        train_tuples = [t for t in all_tuples if t not in test_set]
    train_set = random.sample(train_tuples, int(args.n_train_tasks) )
    print(train_set, test_set)

np.random.seed(SEED)
random.seed(SEED)

# Generate all possible sequences of size d
if args.select_sequences == 1:
    all_sequences = list(itertools.product([0, 1], repeat=d))
    # Randomly select m sequences
    selected_sequences = random.sample(all_sequences, 2**args.n_train_seqs_exponent )
    # Get the remaining sequences
    remaining_sequences = [seq for seq in all_sequences if seq not in selected_sequences]
    remaining_sequences = random.sample(remaining_sequences, 2**args.n_test_seqs_exponent )
else:
    selected_sequences = None
    remaining_sequences = None


if args.cot:

    n_train_per_task = args.n_train_total//len(train_set)

    val_out_diff_seq_dataset = create_dataset(
        d=d, train_task_keys=test_set, N=N, use_cot=True, selected_samples=remaining_sequences, num_samples= 500
    )

    train_dataset = create_dataset(
        d=d, train_task_keys=train_set, N=N, use_cot=True, selected_samples=selected_sequences, num_samples= n_train_per_task
    )



else:

    train_dataset = create_dataset(
        d=d, train_task_keys=train_set, N=N, use_cot=False, selected_samples = selected_sequences
    )
    val_in_dataset = create_dataset(
        d=d, train_task_keys=train_set, N=N, use_cot=False, selected_samples = remaining_sequences
    )

    val_out_same_seq_dataset = create_dataset(
        d=d, train_task_keys=test_set, N=N, use_cot=False, selected_samples = selected_sequences
    )

    val_out_diff_seq_dataset = create_dataset(
        d=d, train_task_keys=test_set, N=N, use_cot=False, selected_samples = remaining_sequences
    )






train_size = len(train_set)

run.config.update({
    "test_set": test_set,
    "context_length": N,
    "train_set": train_set,
    "n_train_tasks": train_size,
    "train_seqs":selected_sequences,
    "test_seqs":remaining_sequences,
    "n_train_total":args.n_train_total,
})
name_data = f"d:{d}--k:{k} --n_train_tasks:{train_size} -- context_length = {N}"




dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) #default 64
dataloader_val_out_diff_seq_dataset = DataLoader(val_out_diff_seq_dataset, batch_size=64, shuffle=True)
print("data is ready")


# Define GPT-2 configuration with small vocabulary and your desired sequence length





config = GPT2Config(
    vocab_size=2,  # Binary tokens: 0 and 1
    n_embd=args.gpt2_width,    # Embedding dimension (default GPT-2)
    n_layer=args.n_layers,    # Number of transformer layers
    n_head=args.n_heads,     # Number of attention heads
    n_positions=2048 # Maximum sequence length
)
# Initialize GPT-2 model
model = GPT2LMHeadModel(config)

if args.resume:
    base_path = "/scratch/bbjr/abedsol1/best_model_d25"
    model.load_state_dict(torch.load(base_path + f"/d25_k3_model_gpt2_context_length40_nlayers3_nheads1_"
                                                 f"ntrain_tasks{args.n_train_tasks}_ntrain_task160n_train_total50000_epoch_600_width192.pt"))


optimizer = AdamW(model.parameters(), lr=args.max_lr)  # default=6e-5# GPT-2 optimizer




name_model = f"model:{args.model_type}_nlayers:{args.n_layers}_nheads:{args.n_heads}"
run.name = name_model + name_data

run.config.update({
    "n_layers":args.n_layers,
    "n_heads":args.n_heads,
    "dim_ambient":args.dim_ambient,
    "dim_k":args.dim_k,
    "gpt2_width":args.gpt2_width,
    "cot":args.cot,
    "data_method":args.data_method,
})




criterion = nn.CrossEntropyLoss(ignore_index = -100)  # For next-token prediction

# optimizer = SGD(model.parameters(), lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
from torch.optim.lr_scheduler import CosineAnnealingLR
epochs = 8000  # Number of training epochs
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) #defualt: eta_min=1e-6
step = 0
accuracy_test_best = 0


for epoch in range(epochs):

    # print(f"epoch{epoch}")
    model.train()
    total_loss = 0
    correct = 0
    n_data = 0

    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
        for batch in progress_bar:

            inputs, targets = batch['input_ids'], batch['labels']
            inputs, targets = inputs.to(device), targets.to(device)
            # Shift targets for next-token prediction
            shifted_targets = targets[:, 1:]  # Ignore the first token
            shifted_inputs = inputs[:, :-1]  # Remove the last token from inputs

            # Forward pass
            outputs = model(input_ids=shifted_inputs)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)



            # Compute loss using reshape instead of view
            loss = criterion(logits.reshape(-1, vocab_size), shifted_targets.reshape(-1))
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.shape[0]

            # Create a mask to identify valid (non-dummy) positions
            logits = torch.argmax(logits, dim=-1)


            # Extract only the last k+1 coordinates
            if args.cot:
                filtered_logits = logits[:, - (k+1):]
                filtered_targets = shifted_targets[:, - (k+1):]
                matches = torch.all(filtered_logits == filtered_targets, dim=1)
            else:
                filtered_logits =  logits[:, - 1:]
                filtered_targets =shifted_targets[:, - 1:]
                matches = torch.all(filtered_logits == filtered_targets, dim=1)


            # Update counters
            n_data += shifted_targets.shape[0]
            correct_count = matches.sum().item()
            correct += correct_count

            # Increment the step counter
            step += 1

    # Step the scheduler after each epoch
    scheduler.step()

    avg_loss = total_loss / n_data
    accuracy_train = correct / n_data
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, accuracy:{accuracy_train}")
    # avg_loss_test, accuracy_test = evaluate_model(model, dataloader_val, criterion, device, k=k)

    if (epoch + 1) % 5 == 0:
        result_out_diff_seq_dataset = evaluate_model(model, dataloader_val_out_diff_seq_dataset, device, k=k, cot = args.cot)

        accuracy_test = result_out_diff_seq_dataset["accuracy_all"]
        print(f"validation Epoch {epoch + 1}/{epochs},  accuracy:{accuracy_test:.4f}")

        current_lr = optimizer.param_groups[0]['lr']

        run.log({ "epoch":epoch,
                 "learning rate": current_lr,
                 "train accuracy": accuracy_train,
                 "train loss":avg_loss,
                 "model":args.model_type,
                  **{f"val_diff_seq_out {key}": value for key, value in result_out_diff_seq_dataset.items()}


                 })


        if accuracy_test_best< accuracy_test and args.save_model and (args.model_save_path is not None):

            accuracy_test_best = accuracy_test
            # Get current day and month
            current_time = datetime.now()
            day = current_time.day
            month = current_time.month
            hour = current_time.hour



            # Create the folder if it does not exist
            os.makedirs(args.save_path, exist_ok=True)

            # Construct the save path within the folder
            save_path = os.path.join(
                args.model_save_path,
                f"d{d}_k{k}_model_{args.model_type}_context_length{N}_nlayers{args.n_layers}"
                f"_nheads{args.n_heads}_ntrain_task{args.n_train_tasks}"
                f"n_train_total{args.n_train_total}_epoch_{epoch + 1}_width{args.gpt2_width}.pt"
            )
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")



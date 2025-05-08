from transformers import set_seed
from tqdm import tqdm
import torch
import numpy as np
import random
import os


def set_all_seeds(seed):
    # Python random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hugging Face Transformers
    set_seed(seed)
    # Environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate_model(model, val_loader,  device, k=3, cot = 0):
    """
    Evaluate the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        val_loader: DataLoader for the validation set.
        criterion: Loss function used during training.
        device: The device ('cpu' or 'cuda') to perform evaluation on.

    Returns:
        result: Dictionary containing average validation loss and accuracies:
                {
                    "avg_loss": Average validation loss,
                    "accuracy_all": Accuracy when all last k+1 coordinates match,
                    "accuracy_0": Accuracy for the last coordinate,
                    "accuracy_1": Accuracy for the second-to-last coordinate,
                    ...
                    "accuracy_k": Accuracy for the k-th coordinate.
                }
    """
    model.eval()  # Set the model to evaluation mode
    correct_all = 0
    correct_coords = [0] * (k + 1)
    total = 0

    with torch.no_grad():  # Disable gradient computation

        for batch in tqdm(val_loader, desc="Validation Progress", unit="batch"):

            inputs, targets = batch['input_ids'], batch['labels']
            inputs, targets = inputs.to(device), targets.to(device)

            shifted_targets = targets[:, 1:]  # Ignore the first token
            inputs = inputs[:, :-1]  # Remove the last token from inputs

            # Forward pass
            outputs = model(input_ids=inputs)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)


            logits = torch.argmax(logits, dim=-1)  # Predicted tokens



            # Compare each coordinate and the full match
            if cot:
                filtered_logits = logits[:, - (k+1):]
                filtered_targets = shifted_targets[:, - (k+1):]
                matches_all = torch.all(filtered_logits == filtered_targets, dim=1)  # All coordinates match
                correct_all += matches_all.sum().item()

                for i in range(k+1 ): #origianlly it was k+1
                    matches_coord = (filtered_logits[:, i] == filtered_targets[:, i])
                    correct_coords[i] += matches_coord.sum().item()

            else:
                filtered_logits = logits[:, - 1:]
                filtered_targets = shifted_targets[:, - 1:]
                matches_all = torch.all(filtered_logits == filtered_targets, dim=1)  # All coordinates match
                correct_all += matches_all.sum().item()

            total += inputs.size(0)

        # Calculate metrics
    # avg_loss = val_loss / total  # Average loss
    accuracy_all = correct_all / total  # Accuracy for all coordinates

    if cot:
        accuracy_coords = [correct / total for correct in correct_coords]

        # Prepare results
        result = { "accuracy_all": accuracy_all}
        for i in range(k + 1):
            result[f"accuracy_{i}"] = accuracy_coords[i]
    else:
        result = {"accuracy_all": accuracy_all}

    return result

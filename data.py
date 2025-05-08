import random
import torch
from torch.utils.data import Dataset
import numpy as np
import ipdb
from tqdm import tqdm

def create_dataset(d, train_task_keys, N, selected_samples=None, use_cot=True, num_samples = None):
    """
    Create a dataset where training and validation use separate sets of task keys.

    :param d: Dimension of the original data.
    :param k: Size of the secret dimension.
    :param train_task_keys: List of task keys for training.
    :param val_task_keys: List of task keys for validation.
    :param n_train: Number of training samples per task key.
    :param n_val: Number of validation samples per task key.
    :param N: Number of sequences concatenated per sample.
    :param use_cot: If True, use chain of thought; otherwise, use final answer only.
    :return: Train and validation datasets as instances of ParityDataset.
    """
    # ipdb.set_trace()
    k = len(train_task_keys[0])
    train_sequences, train_labels, train_masks, train_keys = [], [], [], []

    def generate_samples(task_keys,  N, sequences, labels, masks, keys, selected_samples=None, num_samples = None):
        for secret_coords in tqdm(task_keys, desc="Processing task keys"):
            # Generate all possible sequences of size d
            # all_sequences = np.array([list(format(i, f"0{d}b")) for i in range(2**d)], dtype=int)
            # np.random.shuffle(all_sequences)
            #
            # # Randomly select num_samples from all_sequences
            # selected_samples = all_sequences[:num_samples]

            # if selected_samples is None:
            #     raise ValueError("selected_samples cannot be None.")
            if num_samples is None:
                num_samples = len(selected_samples)


            for point_idx in range(num_samples):
                # Randomly select N sequences without replacement

                # selected_sequences = [np.random.randint(0, 2, size=(1, d), dtype=int).tolist()[0] for _ in range(N)]
                if selected_samples is None:
                    selected_sequences = [np.random.randint(0, 2, size=(1, d), dtype=int).tolist()[0] for _ in range(N)]
                else:
                    selected_sequences = random.sample(list(selected_samples), N)

                # Concatenate the selected sequences
                concatenated_sequence = []
                concatenated_labels = []
                concatenated_mask = []
                for seq in selected_sequences:
                    seq = np.array(seq)

                    # Compute the parity for the secret coordinates
                    parity_values = [seq[coord] for coord in secret_coords]
                    parities = [parity_values[0]]
                    for i in range(1, len(parity_values)):
                        parities.append(parities[-1] ^ parity_values[i])

                    # Add extra coordinates based on CoT or final answer only
                    if use_cot:
                        extra_coordinates = [0] + parities
                    else:
                        extra_coordinates = [0, parities[-1]]

                    # Input sequence with added coordinates
                    input_sequence = np.concatenate([seq, extra_coordinates])

                    # Labels: original -100, added coordinates have the correct labels
                    label = [-100] * d + extra_coordinates

                    # Attention mask (all ones)
                    attention_mask = [1] * len(input_sequence)

                    # Append this sequence to the concatenated result
                    concatenated_sequence.extend(input_sequence)
                    concatenated_labels.extend(label)
                    concatenated_mask.extend(attention_mask)

                # Add the concatenated sequence, labels, and mask
                sequences.append(concatenated_sequence)
                labels.append(concatenated_labels)
                masks.append(concatenated_mask)

                # Generate a meaningful and unique key
                keys.append(f"{secret_coords}")

    # Generate train samples
    generate_samples(train_task_keys, N, train_sequences,
                     train_labels, train_masks, train_keys,selected_samples=selected_samples, num_samples = num_samples)

    # Generate validation samples
    # generate_samples(val_task_keys, n_val, N, val_sequences, val_labels, val_masks, val_keys, "val")

    # Create datasets
    train_dataset = ParityDataset(train_sequences, train_labels, train_masks, train_keys, N, d, k)
    # val_dataset = ParityDataset(val_sequences, val_labels, val_masks, val_keys, N, d, k)

    return train_dataset


class ParityDataset(Dataset):
    def __init__(self, sequences, labels, attention_masks, keys, N, d, k):
        """
        :param sequences: List of concatenated sequences.
        :param labels: List of concatenated labels.
        :param attention_masks: List of concatenated attention masks.
        :param keys: List of keys for debugging.
        :param N: Number of sequences concatenated per input.
        :param d: Dimension of the original data.
        :param k: Size of the secret dimension.
        """
        self.sequences = sequences
        self.labels = labels
        self.attention_masks = attention_masks
        self.keys = keys
        self.N = N
        self.d = d
        self.k = k

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "keys": self.keys[idx],  # Include keys for debugging
        }


if __name__ == '__main__':
    # Configuration
    d = 10  # Dimension of original data
    # k = 2  # Secret dimension size
    N = 3  # Number of sequences concatenated per example
    train_task_keys = [(1,2), (4,2)]  # Training task keys
    val_task_keys = [(7,2)]  # Validation task keys
    n_train = 20  # Number of training samples per task key
    n_val = 15  # Number of validation samples per task key

    # Create datasets
    dataset = create_dataset(
        d=d, train_task_keys=train_task_keys, N=N, use_cot=True, selected_samples=None, num_samples= 500
    )

    # Validate the total number of samples
    print(f"Total train samples: {len(dataset)}")  # Should be len(train_task_keys) * n_train
    print(f"Total validation samples: {len(dataset)}")  # Should be len(val_task_keys) * n_val

    # ipdb.set_trace()


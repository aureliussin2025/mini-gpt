import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

from tokeniser import BPETokenizer


def prepare_bpe_datasets(text, context_size=32):
    """Fit BPE on the training text and create windows within each split."""
    if not 1 <= context_size <= 32:
        raise ValueError("context_size must be between 1 and 32")
    train_end = int(len(text) * 0.70)
    val_end = int(len(text) * 0.85)
    splits = (text[:train_end], text[train_end:val_end], text[val_end:])
    tokenizer = BPETokenizer("byte", pre_tokenize=True).train(splits[0])
    datasets = []
    for name, split in zip(("training", "validation", "test"), splits):
        ids = tokenizer.encode(split)
        if len(ids) <= context_size:
            raise ValueError(f"The {name} split is too short for context_size={context_size}")
        contexts, targets = create_X_and_y(ids, context_size)
        datasets.append(TensorDataset(
            torch.tensor(contexts, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        ))
    return tokenizer, tuple(datasets)

#sliding window
def create_X_and_y(encoded_tokens, context_size=3):
    #setting up datasets
    X = []
    y = []

    #implementing sliding window to allow for context
    for i in range(len(encoded_tokens) - context_size):

        window_size = context_size + 1

        #slicing into 4 elements
        window_of_encoded_tokens = encoded_tokens[i:i + window_size]
        
        #further slicing to 3 and 1
        X.append(window_of_encoded_tokens[:-1])
        y.append(window_of_encoded_tokens[-1])

    X = np.array(X)
    y = np.array(y)

    return X, y

def create_datasets(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        shuffle=False
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        shuffle=False
    )

    #turning the data into tensors
    X_train = torch.tensor(X_train).long()
    X_val = torch.tensor(X_val).long()
    X_test = torch.tensor(X_test).long()

    y_train = torch.tensor(y_train).long()
    y_val = torch.tensor(y_val).long()
    y_test = torch.tensor(y_test).long()

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_train_loader(X_train, y_train, batch_size=128):

    train_dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_dataset, train_loader

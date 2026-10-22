from dataset import create_X_and_y, create_datasets, create_train_loader

import torch
import numpy as np

def test_create_X_and_y():
    encoded_tokens = [1, 2, 3, 4, 5, 6]

    X, y = create_X_and_y(encoded_tokens)

    assert X.tolist() == [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ]

    assert y.tolist() == [4, 5, 6]



def test_create_datasets():

    X = np.arange(300).reshape(100, 3)
    y = np.arange(100)

    X_train, X_val, X_test, y_train, y_val, y_test = create_datasets(X, y)

    # Check dataset sizes
    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15

    # Check tensor types
    assert isinstance(X_train, torch.Tensor)
    assert X_train.dtype == torch.long
    assert y_train.dtype == torch.long

    # Check chronological order
    assert y_train.tolist() == list(range(70))
    assert y_val.tolist() == list(range(70, 85))
    assert y_test.tolist() == list(range(85, 100))

def test_create_train_loader():

    X_train = torch.randint(0, 10, (100, 3))
    y_train = torch.randint(0, 10, (100,))

    train_dataset, train_loader = create_train_loader(
        X_train,
        y_train,
        batch_size=32
    )

    assert len(train_dataset) == 100
    assert len(train_loader) == 4

    X_batch, y_batch = next(iter(train_loader))

    assert X_batch.shape == (32, 3)
    assert y_batch.shape == (32,)
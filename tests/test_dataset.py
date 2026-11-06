from dataset import create_X_and_y, create_datasets, create_train_loader

import pytest
import torch
import numpy as np

def test_create_X_and_y():
    X, y = create_X_and_y(list(range(1, 9)), context_size=3)
    assert X.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert y.tolist() == [[2, 3, 4], [5, 6, 7]]
    # Token 8 belongs to an incomplete trailing chunk and is dropped.


def test_stride_32_boundaries():
    X, y = create_X_and_y(list(range(66)), context_size=32)
    assert X.tolist() == [list(range(32)), list(range(32, 64))]
    assert y.tolist() == [list(range(1, 33)), list(range(33, 65))]


@pytest.mark.parametrize("length", [0, 1, 31, 32])
def test_rejects_incomplete_chunk(length):
    with pytest.raises(ValueError, match="complete"):
        create_X_and_y(list(range(length)), context_size=32)


def test_exactly_one_chunk():
    X, y = create_X_and_y(list(range(33)), context_size=32)
    assert X.shape == y.shape == (1, 32)


def test_create_datasets():

    X = np.arange(300).reshape(100, 3)
    y = np.arange(300).reshape(100, 3)

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
    assert y_train.tolist() == y[:70].tolist()
    assert y_val.tolist() == y[70:85].tolist()
    assert y_test.tolist() == y[85:].tolist()

def test_create_train_loader():

    X_train = torch.randint(0, 10, (100, 3))
    y_train = torch.randint(0, 10, (100, 3))

    train_dataset, train_loader = create_train_loader(
        X_train,
        y_train,
        batch_size=32
    )

    assert len(train_dataset) == 100
    assert len(train_loader) == 4

    X_batch, y_batch = next(iter(train_loader))

    assert X_batch.shape == (32, 3)
    assert y_batch.shape == (32, 3)
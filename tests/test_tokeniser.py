"""Integration checks for the BPE training pipeline."""
import importlib
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import prepare_bpe_datasets
from generate import generate
from model import MiniGPT
from tokeniser import BPETokenizer
from train import evaluate, load_checkpoint, save_checkpoint, train_model


def test_split_isolation_and_dimensions():
    # Exact 70/15/15 boundaries with distinct words in each split.
    text = "aa " * 140 + "bb " * 30 + "cc " * 30
    tokenizer, datasets = prepare_bpe_datasets(text, context_size=32)
    assert b"aa" in tokenizer.vocab
    assert b"bb" not in tokenizer.vocab
    assert b"cc" not in tokenizer.vocab
    for split, dataset in zip(
        ("aa " * 140, "bb " * 30, "cc " * 30), datasets
    ):
        ids = tokenizer.encode(split)
        contexts, targets = dataset.tensors
        assert contexts.shape == (len(ids) - 32, 32)
        assert targets.shape == (len(ids) - 32,)
        assert contexts.dtype == targets.dtype == torch.long
        assert contexts[0].tolist() == ids[:32]
        assert targets.tolist() == ids[32:]


def test_short_split_is_rejected():
    with pytest.raises(ValueError, match="too short"):
        prepare_bpe_datasets("hello", 32)


def test_validation_weights_partial_batch():
    model = MiniGPT(10)
    contexts = torch.randint(0, 10, (5, 3))
    targets = torch.tensor([0, 1, 2, 3, 4])
    loader = DataLoader(TensorDataset(contexts, targets), batch_size=2)
    with torch.no_grad():
        expected = torch.nn.functional.cross_entropy(model(contexts), targets).item()
    assert evaluate(model, loader) == pytest.approx(expected, abs=1e-6)
    assert model.training


def test_checkpoint_round_trip(tmp_path):
    tokenizer = BPETokenizer().train("hello world! hello")
    model = MiniGPT(len(tokenizer.vocab), attention_type="multi").eval()
    contexts = torch.tensor([tokenizer.encode("hello")])
    path = tmp_path / "model.pth"
    save_checkpoint(path, model, tokenizer, 32, 2, 1.5)
    restored, restored_tokenizer, metadata = load_checkpoint(path)
    assert restored_tokenizer.vocab == tokenizer.vocab
    assert restored_tokenizer.merges == tokenizer.merges
    text = "hello 新🙂"
    assert restored_tokenizer.decode(restored_tokenizer.encode(text)) == text
    torch.testing.assert_close(restored(contexts), model(contexts))
    assert metadata["context_size"] == 32
    assert metadata["epoch"] == 2
    assert metadata["validation_loss"] == 1.5
    assert metadata["model_config"]["attention_type"] == "multi"


def test_early_stopping_restores_best_weights(tmp_path, monkeypatch):
    import train

    tokenizer = BPETokenizer().train("ab", vocab_size=256)
    model = MiniGPT(256, attention_type="multi")
    loader = DataLoader(TensorDataset(
        torch.tensor([[97, 98], [98, 97]]), torch.tensor([97, 98])
    ), batch_size=2)
    losses = iter([2.0, 1.0, 1.2, 1.3])
    states = []

    def controlled_validation(current_model, unused_loader):
        states.append({key: value.clone() for key, value in current_model.state_dict().items()})
        return next(losses)

    monkeypatch.setattr(train, "evaluate", controlled_validation)
    path = tmp_path / "best.pth"
    train_model(
        model, loader, loader, tokenizer, context_size=2,
        epochs=10, patience=2, checkpoint_path=path,
    )
    assert len(states) == 4
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[1][key])
    _, _, metadata = load_checkpoint(path)
    assert metadata["epoch"] == 2


def test_generation_keeps_ids_and_crops_context(monkeypatch):
    tokenizer = BPETokenizer().train("ab ", vocab_size=256)
    model = MiniGPT(256, attention_type="multi")
    seen = []

    def fixed_prediction(context):
        seen.append(context.clone())
        logits = torch.full((1, 256), float("-inf"))
        logits[0, ord("b")] = 0
        return logits

    monkeypatch.setattr(model, "forward", fixed_prediction)
    assert generate("a", 35, model, tokenizer) == "a" + "b" * 35
    assert seen[0].shape == (1, 1)
    assert seen[-1].shape == (1, 32)
    assert seen[-1].tolist() == [[ord("b")] * 32]
    assert model.training
    with pytest.raises(ValueError, match="prompt"):
        generate("", 1, model, tokenizer)
    with pytest.raises(ValueError, match="temperature"):
        generate("a", 1, model, tokenizer, temperature=0)


def test_byte_decode_error_policy_and_generated_invalid_bytes(monkeypatch):
    tokenizer = BPETokenizer().train("a", vocab_size=256)
    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode([255])
    assert tokenizer.decode([255], errors="replace") == "�"
    model = MiniGPT(256)

    def invalid_byte(context):
        logits = torch.full((1, 256), float("-inf"))
        logits[0, 255] = 0
        return logits

    monkeypatch.setattr(model, "forward", invalid_byte)
    assert generate("a", 1, model, tokenizer) == "a�"


def test_importing_main_does_not_read_data_train_or_save(monkeypatch):
    import builtins
    from pathlib import Path
    import train

    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not execute the pipeline")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    monkeypatch.setattr(train, "train_model", forbidden)
    sys.modules.pop("main", None)
    importlib.import_module("main")

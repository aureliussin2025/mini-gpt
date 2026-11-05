import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmark_model import forward_time_ms, validation_loss
from generate import generate
from model import MiniGPT
from tokeniser import BPETokenizer
from train import evaluate, load_checkpoint, select_device, train_model


def test_device_selection_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device().type == "cpu"
    assert select_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        select_device("cuda")


def test_device_selection_with_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device().type == "cuda"
    assert select_device("cpu").type == "cpu"
    with pytest.raises(ValueError):
        select_device("invalid")


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA GPU unavailable"
    )),
])
def test_training_loading_generation_and_timing_on_device(device, tmp_path):
    tokenizer = BPETokenizer().train("ab", vocab_size=256)
    model = MiniGPT(256, attention_type="multi").to(device)
    contexts = torch.tensor([[97, 98], [98, 97]])
    loader = DataLoader(TensorDataset(contexts, torch.tensor([97, 98])), batch_size=2)
    path = tmp_path / "device.pth"
    train_model(model, loader, loader, tokenizer, 2, 1, 1, path)
    assert next(model.parameters()).device.type == device
    assert evaluate(model, loader) == pytest.approx(validation_loss(model, loader))
    assert forward_time_ms(model, contexts, repeats=2, warmup=1) > 0
    restored, saved_tokenizer, metadata = load_checkpoint(path, device=device)
    assert next(restored.parameters()).device.type == device
    assert generate("a", 2, restored, saved_tokenizer, 2).startswith("a")
    cpu_model, _, _ = load_checkpoint(path)
    assert next(cpu_model.parameters()).device.type == "cpu"
    with torch.no_grad():
        torch.testing.assert_close(
            cpu_model(contexts), restored(contexts.to(device)).cpu(),
            atol=1e-4, rtol=1e-4,
        )

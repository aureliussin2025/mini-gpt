import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmark_model import forward_time_ms, validation_loss
from model import MiniGPT


def test_validation_loss_weights_partial_batch_and_restores_mode():
    torch.manual_seed(42)
    model = MiniGPT(10)
    contexts = torch.randint(0, 10, (5, 3))
    targets = torch.tensor([1, 2, 3, 4, 5])
    loader = DataLoader(TensorDataset(contexts, targets), batch_size=2)
    with torch.no_grad():
        expected = torch.nn.functional.cross_entropy(model(contexts), targets).item()
    actual = validation_loss(model, loader)
    assert abs(actual - expected) < 1e-6
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_forward_timing_does_not_update_weights_or_mode():
    model = MiniGPT(10, attention_type="multi")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    latency = forward_time_ms(model, torch.zeros(2, 3, dtype=torch.long), repeats=2)
    assert latency > 0
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])

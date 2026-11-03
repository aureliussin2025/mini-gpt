import pytest
import torch

from model import MiniGPT, MultiHeadAttention, SingleHeadAttention


@pytest.mark.parametrize("attention_class", [SingleHeadAttention, MultiHeadAttention])
def test_attention_shape_and_causal_mask(attention_class):
    torch.manual_seed(42)
    attention = attention_class()
    inputs = torch.randn(2, 5, 32)
    changed = inputs.clone()
    changed[:, 3:] = torch.randn(2, 2, 32) * 10
    with torch.no_grad():
        original = attention(inputs)
        modified = attention(changed)
    assert original.shape == inputs.shape
    assert torch.isfinite(original).all()
    torch.testing.assert_close(original[:, :3], modified[:, :3])


@pytest.mark.parametrize("attention_type", ["single", "multi"])
@pytest.mark.parametrize("sequence_length", [1, 3, 32])
def test_model_output_and_gradients(attention_type, sequence_length):
    model = MiniGPT(17, attention_type=attention_type)
    tokens = torch.randint(0, 17, (2, sequence_length))
    logits = model(tokens)
    assert logits.shape == (2, 17)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_multihead_matches_explicit_independent_heads():
    torch.manual_seed(42)
    attention = MultiHeadAttention()
    inputs = torch.randn(2, 3, 32)
    queries = attention.W_q(inputs)
    keys = attention.W_k(inputs)
    values = attention.W_v(inputs)
    mask = torch.ones(3, 3, dtype=torch.bool).tril()
    heads = []
    for head in range(4):
        start = head * 8
        end = start + 8
        scores = queries[..., start:end] @ keys[..., start:end].transpose(-2, -1)
        scores = (scores / (8 ** 0.5)).masked_fill(~mask, float("-inf"))
        heads.append(torch.softmax(scores, dim=-1) @ values[..., start:end])
    expected = attention.output_projection(torch.cat(heads, dim=-1))
    torch.testing.assert_close(attention(inputs), expected)


def test_attention_selection_and_invalid_configuration():
    assert isinstance(MiniGPT(10).attention_block, SingleHeadAttention)
    multi = MiniGPT(10, attention_type="multi").attention_block
    assert isinstance(multi, MultiHeadAttention)
    assert multi.num_heads == 4
    assert multi.head_dim == 8
    with pytest.raises(ValueError):
        MiniGPT(10, attention_type="invalid")
    with pytest.raises(ValueError):
        MultiHeadAttention(32, num_heads=3)

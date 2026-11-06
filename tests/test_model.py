import pytest
import torch

from model import MiniGPT, MultiHeadAttention, SingleHeadAttention, TransformerBlock


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
    assert logits.shape == (2, sequence_length, 17)
    targets = torch.randint(0, 17, (2, sequence_length))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 17), targets.reshape(-1))
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
    assert isinstance(MiniGPT(10).blocks[0].attention_block, SingleHeadAttention)
    multi = MiniGPT(10, attention_type="multi").blocks[0].attention_block
    assert isinstance(multi, MultiHeadAttention)
    assert multi.num_heads == 4
    assert multi.head_dim == 8
    with pytest.raises(ValueError):
        MiniGPT(10, attention_type="invalid")
    with pytest.raises(ValueError):
        MultiHeadAttention(32, num_heads=3)


def test_four_independent_blocks_execute_in_order():
    model = MiniGPT(10, attention_type="multi")
    assert len(model.blocks) == 4
    parameter_sets = [{id(p) for p in block.parameters()} for block in model.blocks]
    for index, parameters in enumerate(parameter_sets):
        for other in parameter_sets[index + 1:]:
            assert parameters.isdisjoint(other)
    visited = []
    handles = [
        block.register_forward_hook(
            lambda module, inputs, output, index=index: visited.append(index)
        )
        for index, block in enumerate(model.blocks)
    ]
    model(torch.tensor([[1, 2, 3]]))
    for handle in handles:
        handle.remove()
    assert visited == [0, 1, 2, 3]


@pytest.mark.parametrize("attention_type", ["single", "multi"])
def test_stacked_blocks_remain_causal(attention_type):
    blocks = torch.nn.Sequential(*[TransformerBlock(attention_type) for _ in range(4)])
    inputs = torch.randn(2, 6, 32)
    changed = inputs.clone()
    changed[:, 3:] += 10
    with torch.no_grad():
        original = blocks(inputs)
        modified = blocks(changed)
    assert original.shape == inputs.shape
    torch.testing.assert_close(original[:, :3], modified[:, :3])


@pytest.mark.parametrize("num_blocks", [0, -1, True, 1.5])
def test_invalid_block_count(num_blocks):
    with pytest.raises(ValueError, match="num_blocks"):
        MiniGPT(10, num_blocks=num_blocks)


@pytest.mark.parametrize("attention_type", ["single", "multi"])
def test_all_position_logits_are_causal(attention_type):
    torch.manual_seed(42)
    model = MiniGPT(17, attention_type=attention_type).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = torch.tensor([[1, 2, 3, 7, 8, 9]])
    with torch.no_grad():
        original_logits = model(tokens)
        changed_logits = model(changed)
    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3])

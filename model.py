import torch
import torch.nn as nn
import math


class SingleHeadAttention(nn.Module):
    """The original causal attention calculation, in its own module."""

    def __init__(self, embedding_dim=32):
        super().__init__()
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(queries.shape[-1])

        sequence_length = x.shape[1]
        mask = torch.ones(
            sequence_length, sequence_length, device=x.device, dtype=torch.bool
        ).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ values


class MultiHeadAttention(nn.Module):
    """Split Q/K/V across heads, then combine their attention outputs."""

    def __init__(self, embedding_dim=32, num_heads=4):
        super().__init__()

        #must be a valid number of heads
        if num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by positive num_heads")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.shape

        queries = self.W_q(x).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = self.W_k(x).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = self.W_v(x).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = torch.ones(
            sequence_length, sequence_length, device=x.device, dtype=torch.bool
        ).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ values

        # Concatenate heads back to (B, T, 32), then mix their outputs.
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch_size, sequence_length, embedding_dim
        )
        return self.output_projection(attended)


class TransformerBlock(nn.Module):
    """Pre-normalised attention and MLP, each with a residual connection."""

    def __init__(self, attention_type="multi"):
        super().__init__()
        self.ln1 = nn.LayerNorm(32)
        self.ln2 = nn.LayerNorm(32)
        if attention_type == "single":
            self.attention_block = SingleHeadAttention(32)
        elif attention_type == "multi":
            self.attention_block = MultiHeadAttention(32, num_heads=4)
        else:
            raise ValueError("attention_type must be 'single' or 'multi'")
        self.mlp = nn.Sequential(
            nn.Linear(32, 100),
            nn.ReLU(),
            nn.Linear(100, 32),
        )

    def forward(self, x):
        x = x + self.attention_block(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, attention_type="single", num_blocks=4):
        super().__init__()
        if isinstance(num_blocks, bool) or not isinstance(num_blocks, int) or num_blocks < 1:
            raise ValueError("num_blocks must be a positive integer")
        self.attention_type = attention_type
        self.embedding = nn.Embedding(vocab_size, 32)
        self.position_embedding = nn.Embedding(32, 32)

        # Construct a new block each time: weights are not shared across layers.
        self.blocks = nn.ModuleList([
            TransformerBlock(attention_type) for _ in range(num_blocks)
        ])
        self.output = nn.Linear(32, vocab_size)

    def forward(self, tokens):
        sequence_length = tokens.shape[1]
        positions = torch.arange(sequence_length, device=tokens.device)
        x = self.embedding(tokens) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        # Each position predicts the next token using only its causal context.
        return self.output(x)

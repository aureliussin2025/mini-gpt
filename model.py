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


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, attention_type="single"):
        super().__init__()

        """
        Embed the words in the vocab (all possible tokens)
        give each token a distribution of 32 initially random number that are trainable
        by back prop
        """
        self.embedding = nn.Embedding(vocab_size, 32)

        """
        positional embedding
        give each of the 32 token positions another number(that can be learnt)
        this is to differentiate tokens by position
        """
        self.position_embedding = nn.Embedding(32, 32)

        """
        layer normalisation
        this normalises the number for stable training

        ln1 is for before attention
        ln2 is before mlp
        """
        self.ln1 = nn.LayerNorm(32)
        self.ln2 = nn.LayerNorm(32)

        if attention_type == "single":
            self.attention_block = SingleHeadAttention(32)
        elif attention_type == "multi":
            self.attention_block = MultiHeadAttention(32, num_heads=4)
        else:
            raise ValueError("attention_type must be 'single' or 'multi'")

        #32 splits to 100 and back to 32
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(32, 100),
            nn.ReLU(),
            nn.Linear(100, 32)
        )

        """
        Convert the final 32-dimensional representation into
        one prediction logit for each token in the vocabulary.
        """
        # Final prediction layer
        self.output = nn.Linear(32, vocab_size)

    def forward(self, tokens):
        #B, batch size. T, tokens per sequenec
        B, T = tokens.shape

        #generate positions from 0 to T-1
        positions = torch.arange(T, device=tokens.device)

        x = self.embedding(tokens) + self.position_embedding(positions)

        x = x + self.attention_block(self.ln1(x))

        x = x + self.mlp(self.ln2(x))

        # Last token's representation predicts the next token
        x = x[:, -1, :]

        # Convert it into prediction scores for every vocabulary token
        logits = self.output(x)

        return logits
        

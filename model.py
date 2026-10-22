import torch.nn as nn

#neural network
def create_model(vocab_size):

    model = nn.Sequential(
        nn.Embedding(vocab_size,32),
        nn.Flatten(start_dim=1),
        nn.Linear(32 * 3, 100), # 3 represents the context size
        nn.ReLU(),
        nn.Linear(100, vocab_size)
    )

    return model
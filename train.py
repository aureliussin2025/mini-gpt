import math
from pathlib import Path

import torch
from torch import nn

from model import MiniGPT, MultiHeadAttention
from tokeniser import BPETokenizer


DEFAULT_CHECKPOINT = Path(__file__).parent / "best_bpe_transformer.pth"


def select_device(name="auto"):
    """Prefer CUDA when available; fail clearly if explicitly requested."""
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Enable a GPU runtime or use --device cpu.")
    return torch.device(name)


def evaluate(model, loader):
    """Compute sample-weighted cross-entropy without modifying model weights."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    count = 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for contexts, targets in loader:
            contexts, targets = contexts.to(device), targets.to(device)
            total_loss += nn.functional.cross_entropy(
                model(contexts), targets, reduction="sum"
            ).item()
            count += targets.numel()
    model.train(was_training)
    if count == 0:
        raise ValueError("Evaluation loader must not be empty")
    return total_loss / count


def save_checkpoint(path, model, tokenizer, context_size, epoch, val_loss):
    attention_type = (
        "multi" if isinstance(model.attention_block, MultiHeadAttention) else "single"
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": len(tokenizer.vocab),
            "attention_type": attention_type,
        },
        "context_size": context_size,
        "tokenizer": {
            "representation": tokenizer.representation,
            "pre_tokenize": tokenizer.pre_tokenize,
            "vocab": tokenizer.vocab,
            "merges": tokenizer.merges,
        },
        "epoch": epoch,
        "validation_loss": val_loss,
    }, path)


def load_checkpoint(path=DEFAULT_CHECKPOINT, device="cpu"):
    """Load portably on CPU, then move the model to the selected device."""
    device = select_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = MiniGPT(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    state = checkpoint["tokenizer"]
    tokenizer = BPETokenizer(state["representation"], state["pre_tokenize"])
    tokenizer.vocab = state["vocab"]
    tokenizer.id_to_token = {index: token for token, index in tokenizer.vocab.items()}
    tokenizer.merges = state["merges"]
    tokenizer._trained = True
    metadata = {
        key: checkpoint[key]
        for key in ("model_config", "context_size", "epoch", "validation_loss")
    }
    return model, tokenizer, metadata


def train_model(
    model, train_loader, val_loader, tokenizer, context_size=32,
    epochs=10, patience=3, checkpoint_path=DEFAULT_CHECKPOINT,
):
    """Train with early stopping and restore the best validation weights."""
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty")
    best_loss = float("inf")
    epochs_without_improvement = 0
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    device = next(model.parameters()).device

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for contexts, targets in train_loader:
            contexts, targets = contexts.to(device), targets.to(device)
            optimiser.zero_grad()
            loss = nn.functional.cross_entropy(model(contexts), targets)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite")
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(targets)
            count += len(targets)

        val_loss = evaluate(model, val_loader)
        if not math.isfinite(val_loss):
            raise ValueError("Validation loss is not finite")
        print(
            f"Epoch {epoch} | Train Loss: {total_loss / count:.4f} | "
            f"Validation Loss: {val_loss:.4f}", flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path, model, tokenizer, context_size, epoch, val_loss
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Restored best weights from epoch {checkpoint['epoch']}")
    return model

"""Compare attention variants on a shared BPE dataset and training budget."""

import argparse
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import create_X_and_y
from model import MiniGPT
from tokeniser import BPETokenizer


def validation_loss(model, loader):
    """Average cross-entropy per target, including a smaller final batch."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_targets = 0
    with torch.inference_mode():
        for contexts, targets in loader:
            total_loss += nn.functional.cross_entropy(
                model(contexts), targets, reduction="sum"
            ).item()
            total_targets += targets.numel()
    model.train(was_training)
    if total_targets == 0:
        raise ValueError("Validation requires at least one target")
    return total_loss / total_targets


def forward_time_ms(model, contexts, repeats=100, warmup=10):
    """Mean CPU inference latency per batch, excluding data loading."""
    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    if contexts.device.type != "cpu" or next(model.parameters()).device.type != "cpu":
        raise ValueError("This benchmark measures CPU inference only")
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(contexts)
        start = time.perf_counter()
        for _ in range(repeats):
            model(contexts)
        elapsed = time.perf_counter() - start
    model.train(was_training)
    return elapsed * 1000 / repeats


def prepare_data(text, context_size):
    # Split raw text before fitting BPE or constructing windows to keep the
    # validation text out of tokenizer training and prevent boundary overlap.
    train_end = int(len(text) * 0.70)
    val_end = int(len(text) * 0.85)
    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    tokenizer = BPETokenizer("byte", pre_tokenize=True).train(train_text)
    datasets = []
    for split in (train_text, val_text):
        encoded = tokenizer.encode(split)
        if len(encoded) <= context_size:
            raise ValueError("Text split is too short; increase --max-chars")
        contexts, targets = create_X_and_y(encoded, context_size)
        datasets.append(TensorDataset(
            torch.tensor(contexts, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        ))
    # Final 15% is reserved for testing, not used for model selection.
    return datasets[0], datasets[1], len(tokenizer.vocab)


def benchmark(text, epochs=3, batch_size=128, context_size=3, repeats=100, seed=42):
    train_data, val_data, vocab_size = prepare_data(text, context_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    timing_batch = next(iter(val_loader))[0]
    print(
        f"CPU | threads={torch.get_num_threads()} | seed={seed} | "
        f"epochs={epochs} | context={context_size} | batch={batch_size}\n"
        f"BPE vocabulary={vocab_size} | train windows={len(train_data)} | "
        f"validation windows={len(val_data)}\n"
        f"Timing batch={len(timing_batch)} | warm-up=10 | repeats={repeats}",
        flush=True,
    )
    results = []
    for attention_type in ("single", "multi"):
        torch.manual_seed(seed)
        model = MiniGPT(vocab_size, attention_type=attention_type)
        # A separate generator gives both variants the same shuffled batches,
        # regardless of how many random values model initialisation consumes.
        loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
        initial_loss = validation_loss(model, val_loader)
        final_loss = initial_loss
        for epoch in range(epochs):
            model.train()
            for contexts, targets in loader:
                optimiser.zero_grad()
                loss = nn.functional.cross_entropy(model(contexts), targets)
                loss.backward()
                optimiser.step()
            final_loss = validation_loss(model, val_loader)
            print(
                f"{attention_type}: epoch {epoch + 1}/{epochs}, "
                f"validation loss={final_loss:.4f}", flush=True,
            )
        latency = forward_time_ms(model, timing_batch, repeats=repeats)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        results.append((attention_type, parameters, latency, initial_loss, final_loss))

    print("\nAttention   Parameters   Forward ms/batch   Initial val CE   Final val CE")
    for mode, parameters, latency, initial, final in results:
        print(f"{mode:<10} {parameters:>10,} {latency:>18.3f} {initial:>16.4f} {final:>14.4f}")
    print("One seed/run; timings vary. Multi-head includes an extra output projection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chars", type=int, default=10000)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.max_chars, args.epochs, args.batch_size, args.repeats, args.threads) <= 0:
        parser.error("sample size, epochs, batch size, repeats and threads must be positive")
    if not 1 <= args.context_size <= 32:
        parser.error("context size must be between 1 and 32")
    torch.set_num_threads(args.threads)
    path = Path(__file__).parent / "data" / "input.txt"
    with path.open(encoding="utf-8") as source:
        text = source.read() if args.full_data else source.read(args.max_chars)
    print(f"Dataset: {path}\nCharacters: {len(text):,} (full={args.full_data})", flush=True)
    benchmark(text, args.epochs, args.batch_size, args.context_size, args.repeats)


if __name__ == "__main__":
    main()

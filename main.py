import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from dataset import prepare_bpe_datasets
from generate import generate
from model import MiniGPT
from train import DEFAULT_CHECKPOINT, evaluate, select_device, train_model


def main():
    parser = argparse.ArgumentParser(description="Train the BPE Transformer")
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.max_chars is not None and args.max_chars <= 0:
        parser.error("--max-chars must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    try:
        device = select_device(args.device)
    except ValueError as error:
        parser.error(str(error))

    # Legacy Windows output encodings may not support generated Unicode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    seed = 42
    context_size = 32
    batch_size = 128
    torch.manual_seed(seed)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device: {device} ({device_name})", flush=True)
    print("Preparing BPE and datasets on CPU ...", flush=True)
    path = Path(__file__).parent / "data" / "input.txt"
    with path.open(encoding="utf-8") as source:
        text = source.read() if args.max_chars is None else source.read(args.max_chars)
    scope = "full dataset" if args.max_chars is None else "sample run"
    print(f"{scope}: {len(text):,} characters", flush=True)
    tokenizer, (train_data, val_data, test_data) = prepare_bpe_datasets(
        text, context_size
    )
    print(f"BPE vocabulary: {len(tokenizer.vocab):,}; context: {context_size}", flush=True)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    model = MiniGPT(len(tokenizer.vocab), attention_type="multi").to(device)
    train_model(
        model, train_loader, val_loader, tokenizer, context_size,
        epochs=args.epochs, patience=3, checkpoint_path=args.checkpoint,
    )
    print(f"Test Loss: {evaluate(model, test_loader):.4f}")
    for temperature in (1.0, 0.5, 2.0):
        print(f"Temperature {temperature}:")
        print(generate(
            "I am a", 20, model, tokenizer, context_size, temperature
        ))


if __name__ == "__main__":
    main()

# mini-gpt

Train the four-head Transformer with byte BPE and 32-token contexts:

```powershell
python main.py
```

For a short sample run:

```powershell
python main.py --max-chars 10000 --epochs 2 --checkpoint sample_bpe.pth
```

Text is split sequentially 70/15/15 before fitting BPE on training text only.
The vocabulary targets 2,048 tokens but may be smaller when no pairs remain.
Training uses Adam (0.001), batch size 128, and early stopping with patience 3.
The best validation weights are restored before one test-set evaluation.
Generation counts new BPE tokens, not words; malformed generated UTF-8 is
displayed with replacement characters.

The default checkpoint is `best_bpe_transformer.pth`, containing model weights,
model settings, context size, tokenizer vocabulary and merges, epoch, and
validation loss. It does not resume optimiser state. The old `best_model.pth`
format is not supported by the new loader and is left untouched.

Load and generate without training:

```python
from train import load_checkpoint
from generate import generate

model, tokenizer, metadata = load_checkpoint()
print(generate(
    "I am a", 20, model, tokenizer,
    context_size=metadata["context_size"], temperature=1.0,
))
```

Run tests with `python -m pytest`. Importing `main` does not start training.
The separate `benchmark_model.py` retains its fixed-epoch attention comparison.

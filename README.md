# mini-gpt

Train the four-layer, four-head Transformer with byte BPE and 32-token contexts:

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
Training predicts the next token at every position: inputs `[a, b, c]` have
targets `[b, c, d]`. With context 32, input chunks advance by 32 tokens within
each split; incomplete final chunks are dropped without padding. Model outputs
have shape `(batch, sequence_length, vocabulary_size)`. Cross-entropy is averaged
over all target tokens, and generation samples only the final position's logits.

Compared with earlier stride-1, last-token-only runs, each epoch now has fewer
chunks and each chunk supervises 32 predictions. Epoch sizes, losses, and timings
are not directly comparable with those earlier runs. Benchmark timing measures
the full-sequence forward pass and validation loss is cross-entropy per token.
Generation counts new BPE tokens, not words; malformed generated UTF-8 is
displayed with replacement characters.

The default checkpoint is `best_bpe_transformer.pth`, containing model weights,
model settings, context size, tokenizer vocabulary and merges, epoch, and
validation loss. It does not resume optimiser state. The old `best_model.pth`
format is not supported by the new loader and is left untouched.
New checkpoints record `prediction_objective="all_positions"` and
`window_stride=32` (the configured context size). Earlier BPE checkpoints load
with `last_token_only` and stride 1 metadata; their recorded loss retains its
original meaning. Parameter shapes are unchanged by all-position prediction.

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

`MiniGPT` passes embeddings through four independent `TransformerBlock` instances.
Each block contains layer normalisation, causal attention, and a 32 → 100 → 32
MLP, with residual connections around attention and the MLP. Both single-head and
four-head attention options use four blocks by default. `num_blocks` can be set
explicitly when constructing a model. New checkpoints store the block count;
older BPE checkpoints load as one-block models, preserving their predictions.
Training a new four-block model does not expand an old checkpoint automatically.

## Google Colab GPU

Push your latest code to GitHub first. In Colab choose **Runtime > Change runtime
type > GPU**, then run these notebook cells (the `%` and `!` prefixes are for
notebooks, not PowerShell):

```python
!git clone https://github.com/aureliussin2025/mini-gpt.git
%cd /content/mini-gpt
%pip install -r requirements.txt
```

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Start with a sample, then run the full dataset when ready:

```python
!python main.py --device cuda --max-chars 10000 --epochs 2 --checkpoint sample_bpe.pth
```

```python
!python main.py --device cuda
```

Training defaults to `--device auto`: CUDA when available, otherwise CPU.
Explicit `--device cuda` reports an error if CUDA is unavailable; `--device cpu`
forces CPU execution. Models move to the selected device before the optimiser
is created. Data stays in CPU memory and each batch moves to the model's device.
Tokenisation and dataset preparation remain CPU work, so GPU acceleration will
not speed up BPE. A small model or sample may not be faster on GPU.

Benchmark on GPU (CPU remains the benchmark default):

```python
!python benchmark_model.py --device cuda --context-size 32
```

Forward timing synchronises CUDA before and after the timed passes and excludes
CPU-to-GPU input transfer. Compare timings using the same batch/context sizes.

Load a checkpoint onto the GPU for generation:

```python
model, tokenizer, metadata = load_checkpoint(device="cuda")
```

The loader defaults to CPU, so GPU-trained checkpoints also work on a CPU laptop.
Colab runtime files are temporary. Download the checkpoint after training:

```python
from google.colab import files
files.download("best_bpe_transformer.pth")
```

Alternatively, mount Drive before training and save directly to it:

```python
from google.colab import drive
drive.mount("/content/drive")
!python main.py --device cuda --checkpoint /content/drive/MyDrive/best_bpe_transformer.pth
```

Checkpoints restore weights/tokenisation for inference; they do not resume the
optimiser or an interrupted training run. GPU and CPU results need not be bitwise
identical even with the same seed.

See the [PyTorch Colab guide](https://docs.pytorch.org/tutorials/beginner/colab.html)
and [Colab FAQ](https://research.google.com/colaboratory/faq.html).

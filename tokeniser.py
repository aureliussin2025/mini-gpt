from collections import Counter
from pathlib import Path
import re
import time


class BPETokenizer:
    """
    constructor
    the tokeniser takes in a represntation of byte vs char and whether or not it should be pretokenised
    the tokeniser also contains the data merges, vocab, ids and whether it was trained
    it raises an error when inputted representation and pre_tokenisation is invalid
    """
    def __init__(self, representation="byte", pre_tokenize=True):
        if representation not in ("char", "byte"):
            raise ValueError("representation must be 'char' or 'byte'")
        if not isinstance(pre_tokenize, bool):
            raise TypeError("pre_tokenize must be a bool")
        self.representation = representation
        self.pre_tokenize = pre_tokenize
        self.merges = []
        self.vocab = {}
        self.id_to_token = {}
        self._trained = False

    """
    split text into chunks properly
    """
    def _pieces(self, text):
        if self.pre_tokenize:
            return re.findall(r"\w+|[^\w\s]|\s+", text)
        return [text] if text else []

    """
    split and convert text chunk into UTF-8 bytes, if representation == byte
    else split and convert text chunk into characters
    """
    def _symbols(self, piece):
        if self.representation == "byte":
            return [bytes([value]) for value in piece.encode("utf-8")]
        return list(piece)

    """
    counting pairs
    """
    @staticmethod
    def _count_pairs(sequences):
        counts = Counter()
        for symbols, frequency in sequences:
            for pair in zip(symbols, symbols[1:]):
                counts[pair] += frequency
        return dict(counts)

    """
    merging pairs
    """
    @staticmethod
    def _merge_pair(symbols, pair):
        # Count overlapping pairs, but merge non-overlapping occurrences.
        merged = []
        index = 0
        while index < len(symbols):
            if (index + 1 < len(symbols)
                    and (symbols[index], symbols[index + 1]) == pair):
                merged.append(pair[0] + pair[1])
                index += 2
            else:
                merged.append(symbols[index])
                index += 1
        return merged

    """
    word frequency (optimisation)
    """
    def _sequences(self, text):
        # Repeated words are processed once, weighted by their frequency.
        return [
            (self._symbols(piece), frequency)
            for piece, frequency in Counter(self._pieces(text)).items()
        ]

    def get_pair_counts(self, text):
        """Count initial adjacent pairs before applying any learned merges."""
        return self._count_pairs(self._sequences(text))

    """
    actual tokenisation part        
    """
    def train(self, text, vocab_size=2048):
        """Merge until vocab_size is reached or no adjacent pairs remain.

        The target includes initial symbols (256 in byte mode). It cannot
        be smaller than that initial vocabulary. Previous training is replaced.
        """
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self._trained = False
        self.merges = []
        initial_symbols = (
            [bytes([value]) for value in range(256)]
            if self.representation == "byte"
            else list(dict.fromkeys(text))
        )
        if vocab_size < len(initial_symbols):
            raise ValueError(
                f"vocab_size must be at least {len(initial_symbols)} "
                "to include all initial symbols"
            )
        self.vocab = {symbol: index for index, symbol in enumerate(initial_symbols)}
        sequences = self._sequences(text)

        while len(self.vocab) < vocab_size:
            pair_counts = self._count_pairs(sequences)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(best_pair)
            merged_symbol = best_pair[0] + best_pair[1]
            if merged_symbol not in self.vocab:
                self.vocab[merged_symbol] = len(self.vocab)
            sequences = [
                (self._merge_pair(symbols, best_pair), frequency)
                for symbols, frequency in sequences
            ]

        self.id_to_token = {index: token for token, index in self.vocab.items()}
        self._trained = True
        return self

    """
    encoding same as before
    """
    def encode(self, text):
        """Apply learned rules in order; unseen characters raise KeyError.

        Byte mode can encode any UTF-8 text, even characters unseen in training.
        """
        if not self._trained:
            raise RuntimeError("Call train() before encode()")
        token_ids = []
        for piece in self._pieces(text):
            symbols = self._symbols(piece)
            for pair in self.merges:
                symbols = self._merge_pair(symbols, pair)
            token_ids.extend(self.vocab[symbol] for symbol in symbols)
        return token_ids

    """
    decoding same as before
    """
    def decode(self, token_ids, errors="strict"):

        """Reconstruct text; arbitrary byte IDs may not form valid UTF-8."""
        if not self._trained:
            raise RuntimeError("Call train() before decode()")
        symbols = [self.id_to_token[token_id] for token_id in token_ids]
        if self.representation == "byte":
            return b"".join(symbols).decode("utf-8", errors=errors)
        return "".join(symbols)


def benchmark(max_chars=None):
    """Compare all modes on the same text, excluding file I/O from timings."""

    #benchmark against dataset
    input_path = Path(__file__).parent / "data" / "input.txt"

    with input_path.open(encoding="utf-8") as input_file:
        if max_chars is None:
            text = input_file.read()            
        else:
            text = input_file.read(max_chars)

    #
    print(f"Dataset: {input_path}", flush=True)
    print(f"Characters: {len(text):,} | UTF-8 bytes: {len(text.encode('utf-8')):,}")
    print("Target vocabulary: 2,048 | One timed run per configuration")
    if max_chars is not None:
        print(f"Sample run: first {max_chars:,} characters only")

    results = []
    for representation in ("char", "byte"):
        for pre_tokenize in (False, True):
            label = f"{representation}, pre_tokenize={pre_tokenize}"
            print(f"\nTraining {label} ...", flush=True)
            tokenizer = BPETokenizer(representation, pre_tokenize)

            start = time.perf_counter()
            tokenizer.train(text, vocab_size=2048)
            train_seconds = time.perf_counter() - start

            print(f"Training finished in {train_seconds:.3f}s; encoding ...", flush=True)
            start = time.perf_counter()
            token_ids = tokenizer.encode(text)
            encode_seconds = time.perf_counter() - start

            start = time.perf_counter()
            decoded = tokenizer.decode(token_ids)
            decode_seconds = time.perf_counter() - start
            if decoded != text:
                raise AssertionError(f"Round-trip failed for {label}")

            results.append((
                label, train_seconds, encode_seconds, decode_seconds,
                len(tokenizer.vocab), len(token_ids),
            ))
            print(f"Vocabulary: {len(tokenizer.vocab):,} | Merges: {len(tokenizer.merges):,}")
            print(f"Encoded tokens: {len(token_ids):,} | Exact round-trip: True")
            print("First 20 IDs:", token_ids[:20])
            print("First 20 tokens:", [tokenizer.id_to_token[i] for i in token_ids[:20]])
            print("Decoded preview:", repr(decoded[:120]), flush=True)

    print("\nSummary (seconds)")
    print(f"{'Mode':<27} {'Train':>10} {'Encode':>10} {'Decode':>10} {'Vocab':>8} {'Tokens':>12}")
    for label, training, encoding, decoding, vocab_size, token_count in results:
        print(
            f"{label:<27} {training:>10.3f} {encoding:>10.3f} "
            f"{decoding:>10.3f} {vocab_size:>8,} {token_count:>12,}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark all four BPE modes")
    parser.add_argument(
        "--max-chars", type=int,
        help="Use only the first N characters; default uses the full dataset",
    )
    args = parser.parse_args()
    if args.max_chars is not None and args.max_chars <= 0:
        parser.error("--max-chars must be positive")
    benchmark(max_chars=args.max_chars)




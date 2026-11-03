import pytest

from tokeniser import BPETokenizer


@pytest.mark.parametrize("representation", ["char", "byte"])
@pytest.mark.parametrize("pre_tokenize", [False, True])
def test_round_trip(representation, pre_tokenize):
    text = "hello, hello!  café 中文 🙂\ncan't\tstop."
    tokenizer = BPETokenizer(representation, pre_tokenize)
    tokenizer.train(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(tokenizer.encode("")) == ""


def test_character_pairs_include_spaces_without_pretokenisation():
    tokenizer = BPETokenizer("char", False)
    text = "hello world!"
    assert tokenizer.get_pair_counts(text) == {
        pair: 1 for pair in zip(text, text[1:])
    }


def test_pretokenisation_weights_repeated_words_and_separates_punctuation():
    tokenizer = BPETokenizer("char", True)
    assert tokenizer.get_pair_counts("hello hello world!") == {
        ("h", "e"): 2, ("e", "l"): 2, ("l", "l"): 2,
        ("l", "o"): 2, ("w", "o"): 1, ("o", "r"): 1,
        ("r", "l"): 1, ("l", "d"): 1,
    }


@pytest.mark.parametrize("representation", ["char", "byte"])
def test_merges_do_not_cross_pretoken_boundaries(representation):
    tokenizer = BPETokenizer(representation, True).train("hi hi!")
    tokens = [tokenizer.id_to_token[i] for i in tokenizer.encode("hi hi!")]
    expected = ["hi", " ", "hi", "!"]
    if representation == "byte":
        expected = [token.encode("utf-8") for token in expected]
    assert tokens == expected


def test_recounts_after_merging_and_breaks_ties_by_first_occurrence():
    tokenizer = BPETokenizer("char", False).train("abab", vocab_size=4)
    assert tokenizer.merges == [("a", "b"), ("ab", "ab")]
    assert len(tokenizer.encode("abab")) == 1
    tied = BPETokenizer("char", False).train("abc", vocab_size=4)
    assert tied.merges == [("a", "b")]


def test_overlapping_pairs_merge_left_to_right():
    tokenizer = BPETokenizer("char", False)
    assert tokenizer.get_pair_counts("aaa") == {("a", "a"): 2}
    tokenizer.train("aaa", vocab_size=2)
    assert [tokenizer.id_to_token[i] for i in tokenizer.encode("aaa")] == ["aa", "a"]


@pytest.mark.parametrize("pre_tokenize", [False, True])
def test_bytes_are_utf8_units_and_support_unseen_characters(pre_tokenize):
    tokenizer = BPETokenizer("byte", pre_tokenize)
    assert tokenizer.get_pair_counts("é") == {(b"\xc3", b"\xa9"): 1}
    tokenizer.train("hello")
    assert tokenizer.decode(tokenizer.encode("新🙂")) == "新🙂"


def test_character_mode_rejects_unseen_characters():
    tokenizer = BPETokenizer("char").train("hello", 10)
    with pytest.raises(KeyError):
        tokenizer.encode("新")


@pytest.mark.parametrize("representation", ["char", "byte"])
def test_empty_training_and_zero_merges(representation):
    tokenizer = BPETokenizer(representation).train("")
    assert tokenizer.merges == []
    assert tokenizer.encode("") == []
    initial_size = 256 if representation == "byte" else 3
    tokenizer.train("abc", vocab_size=initial_size)
    assert len(tokenizer.vocab) == initial_size
    assert tokenizer.merges == []
    assert len(tokenizer.encode("abc")) == 3


def test_retraining_resets_merges_and_vocabulary():
    tokenizer = BPETokenizer("char", False).train("abab", vocab_size=4)
    tokenizer.train("xy", vocab_size=2)
    assert tokenizer.merges == []
    assert tokenizer.vocab == {"x": 0, "y": 1}


def test_invalid_configuration_and_untrained_use():
    with pytest.raises(ValueError):
        BPETokenizer("invalid")
    with pytest.raises(TypeError):
        BPETokenizer(pre_tokenize="yes")
    tokenizer = BPETokenizer()
    with pytest.raises(RuntimeError):
        tokenizer.encode("hello")
    with pytest.raises(RuntimeError):
        tokenizer.decode([])
    with pytest.raises(ValueError):
        tokenizer.train("hello", -1)
    with pytest.raises(TypeError):
        tokenizer.train("hello", 1.5)


@pytest.mark.parametrize("representation", ["char", "byte"])
@pytest.mark.parametrize("pre_tokenize", [False, True])
def test_stops_at_target_vocabulary_size(representation, pre_tokenize):
    initial_size = 256 if representation == "byte" else 4
    tokenizer = BPETokenizer(representation, pre_tokenize)
    tokenizer.train("abcdabcd", vocab_size=initial_size + 2)
    assert len(tokenizer.vocab) == initial_size + 2
    assert tokenizer.decode(tokenizer.encode("abcdabcd")) == "abcdabcd"


@pytest.mark.parametrize("representation", ["char", "byte"])
def test_stops_when_pairs_run_out(representation):
    tokenizer = BPETokenizer(representation).train("a!")
    assert tokenizer.merges == []
    assert len(tokenizer.vocab) < 2048


@pytest.mark.parametrize("representation,target", [("char", 2), ("byte", 255)])
def test_rejects_target_smaller_than_initial_vocabulary(representation, target):
    with pytest.raises(ValueError, match="initial symbols"):
        BPETokenizer(representation).train("abc", vocab_size=target)


def test_default_target_is_2048():
    # 2,047 distinct characters plus one merge reaches the default cheaply.
    text = "".join(chr(0x4E00 + index) for index in range(2047))
    tokenizer = BPETokenizer("char", False).train(text)
    assert len(tokenizer.vocab) == 2048
    assert len(tokenizer.merges) == 1
    assert tokenizer.decode(tokenizer.encode(text)) == text

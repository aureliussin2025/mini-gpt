from tokeniser import tokenise, create_vocab, encode, decode
from dataset import create_X_and_y, create_datasets, create_train_loader
import pytest

#unit testing - tokeniser
def test_tokeniser_no_punctuation():
    text = "hi I am John"
    token = tokenise(text)

    assert token == ["hi", "I", "am", "John"]

def test_tokeniser_space():
    text = "    hi I am     John       "
    token = tokenise(text)

    assert token == ["hi", "I", "am", "John"]

def test_tokeniser_with_punctuation():
    text = "hi, I'm John."
    token = tokenise(text)

    assert token == ["hi", ",", "I'm", "John", "."]

#not solved yet
def test_tokeniser_with_punctuation_and_quotation():
    text = '"hi john"'
    token = tokenise(text)

    assert token == ['"', "hi", "john", '"']

#create_vocab
def test_create_vocab():
    tokens = ["I", "am", "I", "John"]

    word_to_id, id_to_word = create_vocab(tokens)

    assert word_to_id == {
        "I": 0,
        "am": 1,
        "John": 2
    }

    assert id_to_word == {
        0: "I",
        1: "am",
        2: "John"
    }

def test_create_vocab_empty():
    word_to_id, id_to_word = create_vocab([])

    assert word_to_id == {}
    assert id_to_word == {}

#encode/decode
def test_encoding():
    word_to_id = {
        "I": 0,
        "am": 1,
        "John": 2
    }

    result = encode(["I", "am", "John"], word_to_id)

    assert result == [0, 1, 2]


def test_decoding():
    id_to_word = {
        0: "I",
        1: "am",
        2: "John"
    }

    result = decode([0, 1, 2], id_to_word)

    assert result == "I am John "

def test_encoding_unknown_word():
    word_to_id = {"hello": 0}

    with pytest.raises(KeyError):
        encode(["goodbye"], word_to_id)

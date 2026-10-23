import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import string 
import numpy as np

from sklearn.model_selection import train_test_split

from tokeniser import tokenise, create_vocab, encode, decode
from generate import generate#, generate_fixed
from dataset import create_X_and_y, create_datasets, create_train_loader
from model import create_model
from train import train_model

#seed implementation, to keep things repeatable
SEED = 42
torch.maunual_seed(SEED)

#reading data
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

#tokenising text, split text into a list of words
tokens = tokenise(text)

#creating vocab_list
word_id_dict, id_word_dict = create_vocab(tokens)

#encoding the text, convert list of words into ids
encoded_tokens = encode(tokens, word_id_dict)

#creating datasets
X, y = create_X_and_y(encoded_tokens)

#splitting datasets
X_train, X_val, X_test, y_train, y_val, y_test = create_datasets(X, y)

#creating th nn model
vocab_size = len(word_id_dict)
model = create_model(vocab_size)

#training the model
train_dataset, train_loader = create_train_loader(X_train, y_train, batch_size=128)

train_model(
    model,
    train_dataset,
    train_loader,
    X_val, 
    y_val, 
    epochs=10,
    patience=3
)


print("I in vocabulary:", "I" in word_id_dict)
print("i in vocabulary:", "i" in word_id_dict)
print("am in vocabulary:", "am" in word_id_dict)
print("a in vocabulary:", "a" in word_id_dict)

print("Starting tokens:", tokenise("I am a"))
print("Missing tokens:", [
    word for word in tokenise("I am a")
    if word not in word_id_dict
])


#generate text

print(generate("I am a", 20, model, word_id_dict, id_word_dict, temperature=1.0))

print(generate("I am a", 20, model, word_id_dict, id_word_dict, temperature=0.5))

print(generate("I am a", 20, model, word_id_dict, id_word_dict, temperature=2.0))
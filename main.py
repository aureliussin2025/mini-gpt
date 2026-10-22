import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import string 
import numpy as np

from sklearn.model_selection import train_test_split

#refacor - tokenise,eccoding,decoding
"""
further thinking:
i need to create a function to split up the text block properly (puncutation inclusive)

i need to first create the word_id_dict and the id_word dict
through this process, i will have to split up the input

then i will encode the text
this means i will take in a tokens
split it up
tokenise each word

for decoding
the input will be an list of integers
i will need to use id_word_dict, to reverse engineer the integers into words
from there i will join
i must take note that for punctuations, i should remove the space between the 
punctuation and the previous word

this means i need:

split function - this will be known as tokenising
encoding function
decoding function
function to create the word_id_dict (together with the id_word_dict)
"""

#split_txt/toeknise
def tokenise(text):
    token = []
    punctuation_list = "!,?:.;"

    text = text.split()

    for w in text:
        if w[-1] in punctuation_list:
            
            token.append(w[:-1])
            token.append(w[-1:])

        else:
            token.append(w)

    return token


def create_vocab(token):
    id = 0
    word_id_dict = {}

    for w in token:

        if word_id_dict.get(w) == None:
            word_id_dict[w] = id
            id += 1

    id_word_dict = {value: key for key, value in word_id_dict.items()}

    return word_id_dict, id_word_dict

def encode(token):
    encoded_list = []

    for t in token:
        encoded_list.append(word_id_dict[t])

    return encoded_list

def decode(encoded_list):
    decoded_list = ""
    
    for id in encoded_list:
        decoded_list += (id_word_dict.get(id) + " ")
    return decoded_list


#reading data
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

#tokenising text, split text into a list of words
tokens = tokenise(text)

#creating vocab_list
word_id_dict, id_word_dict = create_vocab(tokens)

#encoding the text, convert list of words into ids
encoded_tokens = encode(tokens)

#making the datasets
"""
first, we need to set a context size
say context size = 3
X will need to contain 123, 234, 345,...
y will need to contain 4,   5,   6

possible approach is to loop through encoded_tokens
slice into 4 elements
further slice into 3 and 1
then increment
"""
#setting context_size
context_size = 3

#setting up datasets
X = []
y = []

#implementing sliding window to allow for context
for i in range(len(encoded_tokens) - context_size):

    window_size = context_size + 1

    #slicing into 4 elements
    window_of_encoded_tokens = encoded_tokens[i:i + window_size]
    
    #further slicing to 3 and 1
    X.append(window_of_encoded_tokens[:-1])
    y.append(window_of_encoded_tokens[-1])

X = np.array(X)
y = np.array(y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    shuffle=False
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    shuffle=False
)

#turning the data into tensors
X_train = torch.tensor(X_train).long()
X_val = torch.tensor(X_val).long()
X_test = torch.tensor(X_test).long()

y_train = torch.tensor(y_train).long()
y_val = torch.tensor(y_val).long()
y_test = torch.tensor(y_test).long()

#model info
#upgrade to embedding
vocab_size = len(word_id_dict)
model = nn.Sequential(
    nn.Embedding(vocab_size,32),
    nn.Flatten(start_dim=1),
    nn.Linear(32 * context_size, 100),
    nn.ReLU(),
    nn.Linear(100, vocab_size)
)

loss_fn = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Creating the training dataset
train_dataset = TensorDataset(X_train, y_train)

# Creating the DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

#actual bot function
def generate(start_words, total_num_of_words):
    
    output = [start_words]

    tokenised_start_words = tokenise(start_words)

    model.eval()

    with torch.no_grad():
    
        for i in range(total_num_of_words):

            current_words = torch.tensor([encode(tokenised_start_words)])

            logits = model(current_words)

            probabilities = torch.softmax(logits, dim=-1)

            next_word_id = torch.multinomial(probabilities, num_samples=1)

            next_word = id_word_dict[next_word_id.item()]

            output.append(next_word)

            tokenised_start_words = tokenised_start_words[1:] + [next_word]

    return " ".join(output)

#training loop
for epoch in range(10):

    model.train()

    total_loss = 0

    # Process one batch at a time
    for X_batch, y_batch in train_loader:

        # Forward pass
        prediction = model(X_batch)

        # Calculate loss
        loss = loss_fn(prediction, y_batch)

        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(X_batch)

# Average training loss across the entire epoch
    avg_train_loss = total_loss / len(train_dataset)

    # Validation
    model.eval()

    with torch.no_grad():

        val_prediction = model(X_val)
        val_loss = loss_fn(val_prediction, y_val)

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Validation Loss: {val_loss.item():.4f}"
    )
    print("\n")
    print(generate("I am a", 20))


# print(generate("I am a", 20))

# #generate fixed
# def generate_fixed(start_words, total_num_of_words):
    
#     output = [start_words]

#     tokenised_start_words = tokenise(start_words)

#     model.eval()

#     with torch.no_grad():
    
#         for i in range(total_num_of_words):

#             current_word = torch.tensor(encode(tokenised_start_words))

#             logits = model(current_word)

#             next_word_id = torch.argmax(logits)

#             next_word = id_word_dict[next_word_id.item()]

#             output.append(next_word)

#             tokenised_start_words = tokenised_start_words[1:] + [next_word]

#     return " ".join(output)

# print(generate_fixed("I am a", 4))
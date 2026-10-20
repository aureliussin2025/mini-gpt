import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import string 
import numpy as np

from sklearn.model_selection import train_test_split

#encoding and decoding function
def encode(text):
    res = []

    text = text.split()
    for word in text:
        res.append(word_id_dict.get(word))
    
    return res

def decode(token):
    res = ""

    for id in token:
        res += (id_word_dict.get(id) + " ")
    return res

#reading data
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

#preserving a copy of text
data = text

#split words into text into words
text = text.split()

#assigning words to unique id
id = 0
word_id_dict = {}
for w in text:
    if word_id_dict.get(w) == None:
        word_id_dict[w] = id
    else:
        continue
    
    id += 1

id_word_dict = {value: key for key, value in word_id_dict.items()}

#making the datasets
X = np.array(encode(data)[:-1])
y = np.array(encode(data)[1:])

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
    nn.Linear(32,100),
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

#actual bot function
def generate(start_word, total_num_of_words):
    
    output = [start_word]

    model.eval()

    with torch.no_grad():
    
        for i in range(total_num_of_words):

            current_word = torch.tensor(encode(start_word))

            logits = model(current_word)

            probabilities = torch.softmax(logits, dim=-1)

            next_word_id = torch.multinomial(probabilities, num_samples=1)

            next_word = id_word_dict[next_word_id.item()]

            output.append(next_word)

            start_word = next_word

    return " ".join(output)

print(generate("I", 4))

#generate fixede
def generate_fixed(start_word, total_num_of_words):
    
    output = [start_word]

    model.eval()

    with torch.no_grad():
    
        for i in range(total_num_of_words):

            current_word = torch.tensor(encode(start_word))

            logits = model(current_word)

            next_word_id = torch.argmax(logits)

            next_word = id_word_dict[next_word_id.item()]

            output.append(next_word)

            start_word = next_word

    return " ".join(output)

print(generate_fixed("I", 4))
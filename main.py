import torch
import torch.nn as nn
import string 
import numpy as np

from sklearn.model_selection import train_test_split

#encoding and decoding function
def encode(text):
    res = []
    for c in text:
        res.append(ord(c))
    return res

def decode(token):
    res = ""
    for c in token:
        res += chr(c)
    return res

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

X = np.array(encode(text)[:-1])
y = np.array(encode(text)[1:])

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50
)

X_train = torch.tensor(X_train).float().unsqueeze(1)
X_val = torch.tensor(X_val).float().unsqueeze(1)
X_test = torch.tensor(X_test).float().unsqueeze(1)

y_train = torch.tensor(y_train)
y_val = torch.tensor(y_val)
y_test = torch.tensor(y_test)

model = nn.Sequential(
    nn.Linear(1,100),
    nn.ReLU(),
    nn.Linear(100, 128)
)

loss_fn = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(1000):

    model.train()

    prediction = model(X_train)

    loss = loss_fn(prediction, y_train)

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    # Check validation performance (no learning)
    model.eval()

    with torch.no_grad():
        val_prediction = model(X_val)
        val_loss = loss_fn(val_prediction, y_val)

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Validation Loss: {val_loss.item():.4f}"
        )
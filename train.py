import torch
import torch.nn as nn


def train_model(model, train_dataset, train_loader, X_val, y_val, epochs, patience):

    best_model_validation = float('inf')

    epochs_without_improvement = 0

    loss_fn = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Training loop
    for epoch in range(epochs):

        model.train()
        total_loss = 0

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

        #state saving process
        if best_model_validation > val_loss.item():
            epochs_without_improvement = 0
            best_model_validation = val_loss.item()

            #save state with torch.save
            torch.save(model.state_dict(), "best_model.pth")

            #indicator when there is a new best model
            print(f"New best model saved at Epoch {epoch+1}")
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at Epoch {epoch+1}")
                break



    model.load_state_dict(
        torch.load("best_model.pth", weights_only=True)   
    )

    return model
        
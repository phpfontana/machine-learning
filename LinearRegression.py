# importing libraries
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# Setting hyper parameters
epochs = 200
learning_rate = 0.01

# Dummy dataset
start = 0
end = 1
step = 0.2

X = torch.arange(start, end, step).unsqueeze(dim=1)  # range of values for X

weight = 0.7
bias = 0.3
y = weight * X + bias  # y outputs given X inputs and parameters (weight/bias)

train_split = int(0.8 * len(X))  # defining train/validation split

X_train, y_train = X[:train_split], y[:train_split]  # Training data
X_val, y_val = X[train_split:], y[train_split:]  # Validation data


# Linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining weight parameter
        self.weights = nn.Parameter(torch.randn(1,  # Initialize with random weight parameter
                                                requires_grad=True,  # Requires gradient descent set to `True`
                                                dtype=torch.float))
        # Defining bias parameter
        self.bias = nn.Parameter(torch.randn(1,  # Initialize with random bias parameter
                                             requires_grad=True,  # Requires gradient descent set to `True`
                                             dtype=torch.float))

    # Forward method
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # the input x is of Torch tensor type
        return self.weights * x + self.bias  # Linear regression


# Instance of the model
model = LinearRegression()


# criterion and optimizer
class MAELoss(nn.Module):
    def forward(self, predicted, actual) -> torch.Tensor:
        return torch.mean(torch.abs(predicted - actual))  # MAE loss


criterion = MAELoss()
optimizer = torch.optim.SGD(params=model.parameters(),  # Optimize model's parameters
                            lr=learning_rate)  # lr = Learning Rate

# Tracking different values
epoch_count = []
train_loss_values = []
val_loss_values = []

# Training step
for epoch in range(epochs):  # Loop through the data
    # Set model to training mode
    model.train()

    # Forward pass
    y_train_pred = model(X_train)

    # Loss
    train_loss = criterion(y_train_pred, y_train)

    # Backward prop and optimizer
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()  # Perform gradient descent

    # Evaluation step
    model.eval()  # Set model to evaluation mode
    with torch.inference_mode():

        # Forward pass
        y_val_pred = model(X_val)

        # Loss
        val_loss = criterion(y_val_pred, y_val)

    # Printing training step and results
    if (epoch + 1) % 5 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
        print(f"Epoch: {epoch + 1}/{epochs} | Train loss: {train_loss} | Val loss: {val_loss}")
        print(model.state_dict())
        print("\n")

# Plot loss curve
plt.plot(epoch_count, np.array(torch.tensor(train_loss_values).numpy()), label="Train loss")
plt.plot(epoch_count, np.array(torch.tensor(val_loss_values).numpy()), label="Validation loss")
plt.title("Training and validation loss values")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

# Save the model state dict
torch.save(model.state_dict(), 'linear-regression.pt')

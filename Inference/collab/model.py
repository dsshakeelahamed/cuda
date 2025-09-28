# Vanilla training
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
def plot_example(X, y):
    """Plot the first 5 images and their labels in a row."""
    for i, (img, y) in enumerate(zip(X[:10].reshape(10, 28, 28), y[:10])):
        plt.subplot(151 + i)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(y)

# 1. Load + normalize MNIST
print("Load")
mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
X = mnist.data.astype('float32') / 255.0
y = mnist.target.astype('int64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Convert to torch tensors
print("Convert")
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test)

# 3. Define a very small NN
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNN()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 4. Loss + Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. Training loop
print("Train")
batch_size = 128
for epoch in range(5):  # 5 epochs is enough for >90% accuracy
    model.train()
    perm = torch.randperm(X_train_t.size(0))
    total_loss = 0
    for i in range(0, X_train_t.size(0), batch_size):
        idx = perm[i:i+batch_size]
        xb, yb = X_train_t[idx].to(device), y_train_t[idx].to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 6. Evaluation
print("Eval")
model.eval()
with torch.no_grad():
    t1 = datetime.now()
    preds = model(X_test_t.to(device)).argmax(dim=1)
    t2 = datetime.now()
    print("Time taken:", (t2 - t1).total_seconds())
    acc = (preds.cpu() == y_test_t).float().mean()
    print("Test Accuracy:", acc.item())
    print("Predictions ", preds)

# Save weights row-major
w1 = model.fc1.weight.detach().cpu().numpy().astype(np.float32)  # [128, 784]
b1 = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
w2 = model.fc2.weight.detach().cpu().numpy().astype(np.float32)  # [10, 128]
b2 = model.fc2.bias.detach().cpu().numpy().astype(np.float32)

# IMPORTANT: transpose because PyTorch stores [out_features, in_features]
w1 = w1.T  # [784, 128]
w2 = w2.T  # [128, 10]

w1.tofile("weights1.bin")
b1.tofile("bias1.bin")
w2.tofile("weights2.bin")
b2.tofile("bias2.bin")


# Extract first 5 samples
X_test_np = X_test[:5]  # shape [5, 784]
y_test_np = y_test[:5]  # shape [5]

print("X_test_np shape:", X_test_np.shape)
print("Y_test_np shape:", y_test_np.shape)
# Save to binary files
X_test_np.astype(np.float32).tofile("test_inputs.bin")
y_test_np.astype(np.int32).tofile("test_labels.bin")

print("All process complete")


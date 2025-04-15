import torch
import torch.nn as nn
import torch.optim as optim

# Dummy neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        return self.fc(x)

# Check if multiple GPUs are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Create model and wrap with DataParallel if multiple GPUs
model = SimpleNet()
if num_gpus > 1:
    model = nn.DataParallel(model)
model.to(device)

# Dummy input and target
inputs = torch.randn(64, 1000).to(device)
targets = torch.randint(0, 10, (64,)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training step
outputs = model(inputs)
loss = criterion(outputs, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()

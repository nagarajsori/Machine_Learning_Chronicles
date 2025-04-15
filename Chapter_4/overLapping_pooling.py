import torch
import torch.nn as nn

# Dummy input tensor: (batch_size, channels, height, width)
x = torch.randn(1, 1, 6, 6)

# Overlapping max pooling: kernel_size > stride
pool = nn.MaxPool2d(kernel_size=3, stride=2)
output = pool(x)

print("Input shape:", x.shape)
print("Output shape after overlapping pooling:", output.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dummy input tensor: (batch_size, channels, height, width)
x = torch.randn(1, 5, 7, 7)

# Apply Local Response Normalization
lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
output = lrn(x)

print("Input shape:", x.shape)
print("Output shape after LRN:", output.shape)

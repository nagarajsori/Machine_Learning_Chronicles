# Simulates one iteration of wake-sleep algorithm for a simple 2-layer network

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights
W_gen = np.random.normal(0, 0.1, (4, 6))  # Generative weights (hidden -> visible)
W_rec = W_gen.T                           # Recognition weights (visible -> hidden)

# Wake phase (bottom-up pass)
data = np.array([1, 0, 1, 0, 1, 0])
hidden_up = sigmoid(np.dot(data, W_rec))

# Sleep phase (top-down reconstruction)
recon = sigmoid(np.dot(hidden_up, W_gen))
hidden_down = sigmoid(np.dot(recon, W_rec))

# Update recognition weights using reconstruction difference
lr = 0.1
W_rec += lr * np.outer(recon, hidden_up - hidden_down)

print("Updated recognition weights after wake-sleep step:")
print(W_rec)

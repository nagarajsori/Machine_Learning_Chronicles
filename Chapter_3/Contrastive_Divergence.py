# Apply one iteration of contrastive divergence to update weights and biases

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initial parameters
W = np.random.normal(0, 0.1, (6, 3))
vb = np.zeros(6)
hb = np.zeros(3)

# Input vector
v0 = np.array([1, 0, 1, 0, 1, 0])

# Positive phase
h0_prob = sigmoid(np.dot(v0, W) + hb)
h0_sample = (h0_prob > np.random.rand(*h0_prob.shape)).astype(float)

# Negative phase
v1_prob = sigmoid(np.dot(h0_sample, W.T) + vb)
v1_sample = (v1_prob > np.random.rand(*v1_prob.shape)).astype(float)
h1_prob = sigmoid(np.dot(v1_sample, W) + hb)

# Weight update
learning_rate = 0.1
W += learning_rate * (np.outer(v0, h0_prob) - np.outer(v1_sample, h1_prob))
vb += learning_rate * (v0 - v1_sample)
hb += learning_rate * (h0_prob - h1_prob)

print("Updated weights:\n", W)

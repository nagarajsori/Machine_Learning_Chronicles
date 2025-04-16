# Generates a sample by running Gibbs sampling on a trained RBM

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gibbs_sample(W, vb, hb, steps=1000):
    n_vis = W.shape[0]
    n_hid = W.shape[1]
    v = np.random.randint(0, 2, n_vis)  # Random initial visible state

    for _ in range(steps):
        h_prob = sigmoid(np.dot(v, W) + hb)
        h = (h_prob > np.random.rand(n_hid)).astype(float)
        v_prob = sigmoid(np.dot(h, W.T) + vb)
        v = (v_prob > np.random.rand(n_vis)).astype(float)

    return v

# Set up small associative memory
W = np.random.normal(0, 0.1, (6, 3))
vb = np.zeros(6)
hb = np.zeros(3)

# Generate sample
generated = gibbs_sample(W, vb, hb, steps=1000)
print("Sampled image from model memory:", generated)

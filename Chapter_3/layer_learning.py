# Greedy unsupervised layer-wise training of a DBN using stacked RBMs

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RBM:
    def __init__(self, n_vis, n_hid):
        self.W = np.random.normal(0, 0.1, (n_vis, n_hid))
        self.vb = np.zeros(n_vis)
        self.hb = np.zeros(n_hid)

    def train(self, data, epochs=10, lr=0.1):
        for epoch in range(epochs):
            for v0 in data:
                h_prob = sigmoid(np.dot(v0, self.W) + self.hb)
                h_sample = (h_prob > np.random.rand(*h_prob.shape)).astype(float)
                v_prob = sigmoid(np.dot(h_sample, self.W.T) + self.vb)
                h_recon = sigmoid(np.dot(v_prob, self.W) + self.hb)

                # Update weights
                self.W += lr * (np.outer(v0, h_prob) - np.outer(v_prob, h_recon))
                self.vb += lr * (v0 - v_prob)
                self.hb += lr * (h_prob - h_recon)

    def transform(self, data):
        return sigmoid(np.dot(data, self.W) + self.hb)

# Create synthetic binary data
data = np.random.randint(0, 2, (100, 6))

# Train first RBM (6 -> 4)
rbm1 = RBM(6, 4)
rbm1.train(data)

# Transform data using first RBM
hidden1 = rbm1.transform(data)

# Train second RBM (4 -> 2)
rbm2 = RBM(4, 2)
rbm2.train(hidden1)

print("Greedy layer-wise training completed.")

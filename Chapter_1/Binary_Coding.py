# Simple Perceptron to classify objects using binary features

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, x):
        # Simple step function
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]  # bias + weighted sum
        return self.activation(z)

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                # update weights and bias
                self.weights[1:] += self.lr * error * xi
                self.weights[0] += self.lr * error

# Let's define some objects using binary features:
# Feature1: Dark(1)/Light(0)
# Feature2: Tall(1)/Short(0)
# Feature3: Curved(1)/Straight(0)

# Example data (binary features)
# Each row is [Dark, Tall, Curved]
X = np.array([
    [1, 1, 1],  # Dark, Tall, Curved
    [1, 0, 1],  # Dark, Short, Curved
    [0, 1, 0],  # Light, Tall, Straight
    [0, 0, 0]   # Light, Short, Straight
])

# Labels: 1 = Class A (e.g., "Dog"), 0 = Not Class A
y = np.array([1, 1, 0, 0])

# Create and train perceptron
p = Perceptron(input_size=3)
p.train(X, y)

# Try predicting new inputs
test_input = np.array([1, 1, 0])  # Dark, Tall, Straight
print(f"Prediction for {test_input}: Class", p.predict(test_input))

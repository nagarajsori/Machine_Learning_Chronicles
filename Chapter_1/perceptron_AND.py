import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        # Initialize weights to zeros and set bias to 0
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        # A simple step function — returns 1 if x >= 0, else 0
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Calculate the linear combination of inputs and weights
        linear_output = np.dot(x, self.weights) + self.bias
        # Apply activation to get the binary output
        return self.activation_function(linear_output)

    def fit(self, X, y):
        # Loop through the dataset multiple times (epochs)
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}")
            for i in range(len(X)):
                # Make a prediction
                prediction = self.predict(X[i])
                # Calculate the error (difference between actual and predicted)
                error = y[i] - prediction

                # Update the weights and bias if there's an error
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

                # Log progress
                print(f"  Input: {X[i]} | Target: {y[i]} | Prediction: {prediction} | Error: {error}")
                print(f"  --> Updated weights: {self.weights}, Bias: {self.bias}")
            print()  # Just to separate epochs

# Define inputs and expected outputs for the AND logic gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # Expected outputs

# Create a perceptron and train it
perceptron = Perceptron(input_size=2)
perceptron.fit(X, y)

# Final test: check if it learned the AND logic
print("Final Predictions:")
for x in X:
    result = perceptron.predict(x)
    print(f"Input: {x} --> Output: {result}")

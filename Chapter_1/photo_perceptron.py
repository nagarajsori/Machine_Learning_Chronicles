import numpy as np

class PhotoPerceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=20):
        # Initialize weights and bias to zero
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        # Basic step function: returns 1 if input is positive, else 0
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Dot product + bias gives the raw output (before activation)
        total_input = np.dot(x, self.weights) + self.bias
        return self.activation_function(total_input)

    def fit(self, X, y):
        # Training the model using Perceptron learning rule
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            for i in range(len(X)):
                x_sample = X[i]
                target = y[i]
                prediction = self.predict(x_sample)
                error = target - prediction

                # Update rule: adjust weights and bias if there was an error
                self.weights += self.lr * error * x_sample
                self.bias += self.lr * error

                print(f"  Input Image #{i+1}")
                print(f"  Target: {target}, Predicted: {prediction}, Error: {error}")
                print(f"  Updated Weights: {self.weights}")
                print(f"  Updated Bias: {self.bias}")
    
    def display_image(self, flat_array):
        # Utility function to print the 3x3 image nicely
        print("  Image:")
        for i in range(0, 9, 3):
            print("  ", flat_array[i:i+3])

# Define 3x3 binary images (flattened into 9-element vectors)
X = np.array([
    [0, 1, 0,
     0, 1, 0,
     0, 1, 0],  # Vertical center line (should detect)

    [1, 0, 1,
     0, 0, 0,
     1, 0, 1],  # Diagonal and noise (should NOT detect)

    [0, 0, 0,
     1, 1, 1,
     0, 0, 0],  # Horizontal line (should NOT detect)

    [0, 1, 0,
     0, 1, 0,
     0, 1, 0],  # Same vertical line again
])

# Labels: 1 = pattern exists, 0 = pattern does not exist
y = np.array([1, 0, 0, 1])

# Create and train the perceptron
photo_brain = PhotoPerceptron(input_size=9)
photo_brain.fit(X, y)

# Final Predictions
print("\nFinal Predictions:")
for idx, img in enumerate(X):
    result = photo_brain.predict(img)
    print(f"\nImage #{idx+1}")
    photo_brain.display_image(img)
    print(f"  Predicted Output: {result}")

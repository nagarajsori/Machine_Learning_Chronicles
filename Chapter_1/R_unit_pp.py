import numpy as np

class AUnit:
    """
    Simulates a feature detector. This is a simple perceptron trained to recognize one visual feature.
    """
    def __init__(self, input_size, learning_rate=0.1, epochs=15):
        self.weights = np.zeros(input_size + 1)
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation(z)

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                self.weights[1:] += self.lr * error * xi
                self.weights[0] += self.lr * error

class RUnit:
    """
    This R-unit takes the outputs from A-units and makes the final classification.
    """
    def __init__(self, num_afeatures, learning_rate=0.1, epochs=20):
        self.weights = np.zeros(num_afeatures + 1)
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, a_outputs):
        z = np.dot(a_outputs, self.weights[1:]) + self.weights[0]
        return self.activation(z)

    def train(self, A_outputs, y_labels):
        for _ in range(self.epochs):
            for a_out, label in zip(A_outputs, y_labels):
                pred = self.predict(a_out)
                error = label - pred
                self.weights[1:] += self.lr * error * a_out
                self.weights[0] += self.lr * error

# ----- Simulated photo feature data -----
# Features: [Bright, Symmetric, Sharp Edges]
photos = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 0]
])

# Final target label: 1 = has face, 0 = no face
labels = np.array([1, 1, 0, 0])

# Let's train three A-units, one per feature (they can specialize)
a_units = [AUnit(input_size=3) for _ in range(3)]

# For training simplicity, each A-unit will try to learn to fire for just one feature pattern
# We'll label each A-unit's training target to be its corresponding feature column
for i, au in enumerate(a_units):
    au.train(photos, photos[:, i])  # Supervised to fire for its specific feature

# Get A-unit outputs for all photos
a_unit_outputs = np.array([[au.predict(photo) for au in a_units] for photo in photos])

# Train the R-unit on A-unit outputs
r_unit = RUnit(num_afeatures=3)
r_unit.train(a_unit_outputs, labels)

# ---- Test ----
test_photo = np.array([1, 1, 0])  # Bright, Symmetric, Blurry
a_out_test = np.array([au.predict(test_photo) for au in a_units])
final_prediction = r_unit.predict(a_out_test)

print("Test photo (features):", test_photo)
print("A-unit outputs:", a_out_test)
print("R-unit (final class):", final_prediction)

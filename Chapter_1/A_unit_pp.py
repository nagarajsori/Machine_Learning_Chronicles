import numpy as np

class AUnitPhotoPerceptron:
    def __init__(self, num_features, learning_rate=0.1, epochs=20):
        # One weight per feature + 1 for bias
        self.weights = np.zeros(num_features + 1)
        self.lr = learning_rate
        self.epochs = epochs

    def _activation(self, x):
        return 1 if x >= 0 else 0  # Step function

    def _weighted_sum(self, features):
        return np.dot(features, self.weights[1:]) + self.weights[0]  # bias + sum(w*x)

    def predict(self, features):
        z = self._weighted_sum(features)
        return self._activation(z)

    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for features, label in zip(X, y):
                prediction = self.predict(features)
                error = label - prediction
                total_error += abs(error)
                # Update rule
                self.weights[1:] += self.lr * error * features
                self.weights[0] += self.lr * error  # bias update
            print(f"Epoch {epoch+1}: Total Error = {total_error}")
            if total_error == 0:
                break

# ---- Simulation: Simplified binary features from "photos" ----
# Let's say we extracted these binary features from photos:
# [Bright(1)/Dark(0), HasSymmetry(1)/NoSymmetry(0), EdgeSharpness(1)/Blurry(0)]

X = np.array([
    [1, 1, 1],  # Bright, Symmetric, Sharp → likely a face
    [1, 0, 1],  # Bright, Asymmetric, Sharp
    [0, 1, 0],  # Dark, Symmetric, Blurry
    [0, 0, 0],  # Dark, Asymmetric, Blurry → unlikely to be a face
])

# Labels: 1 = Photo has face, 0 = No face
y = np.array([1, 1, 0, 0])

# Create, train, and test the photo-perceptron
photo_net = AUnitPhotoPerceptron(num_features=3)
photo_net.train(X, y)

# Testing new "photo"
test_photo = np.array([1, 1, 0])  # Bright, Symmetric, Blurry
print("Predicted class for test photo:", photo_net.predict(test_photo))

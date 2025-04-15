import numpy as np
import matplotlib.pyplot as plt

# Define ReLU function
def relu(x):
    return np.maximum(0, x)

# Generate input data
x = np.linspace(-10, 10, 100)
y = relu(x)

# Plot ReLU
plt.plot(x, y, label='ReLU', color='blue')
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()

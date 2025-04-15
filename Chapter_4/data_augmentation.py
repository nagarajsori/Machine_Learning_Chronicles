from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load an example image
img = Image.open('example.jpg')  # Use any local image path

# Define data augmentation transforms
augment = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])

# Apply augmentation
augmented_img = augment(img)

# Display original and augmented images
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(augmented_img)
plt.title("Augmented Image")
plt.show()

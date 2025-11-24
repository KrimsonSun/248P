import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# --- 1. Model Definition (The 6-Line ConvNet Architecture) ---
# A basic ConvNet consists of a stack of Conv2D and MaxPooling2D layers, 
# followed by a Dense classifier.

print("--- Defining the ConvNet Architecture ---")
model = models.Sequential()

# 1. Convolutional Block (Feature Extraction)
# Input: 28x28x1 (height, width, channels)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# Output: 26x26x32 (Filter size 3x3 reduces dimensions by 2)
model.add(layers.MaxPooling2D((2, 2))) # Output: 13x13x32 (Halves dimensions)
# use a 2x2 window to downsample the sample


# 2. Second Convolutional Block
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Output: 11x11x64
model.add(layers.MaxPooling2D((2, 2))) # Output: 5x5x64

# 3. Third Convolutional Block
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Output: 3x3x64

# --- 4. Classifier Block (Classification) ---
# Flatten the 3D tensor output into a 1D vector for the Dense layers.
model.add(layers.Flatten()) # Output: 3*3*64 = 576 units  

##we have to add this layer turn 3d to 1d vector. then we can turn it into dense layer.

# Dense Hidden Layer
model.add(layers.Dense(64, activation='relu'))

# Output Layer: 10 units for 10-way classification, using softmax for probability distribution.
model.add(layers.Dense(10, activation='softmax'))
#For hidden layers usually use relu
#For output use softmax or sigmoid


# Display the model architecture and parameter count
print("\n--- Model Summary ---")
model.summary()

# --- 2. Data Loading and Preprocessing ---
print("\n--- Loading and Preprocessing MNIST Data ---")

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape the images to fit the ConvNet input requirement: (samples, height, width, channels)
# MNIST is grayscale, so the channel is 1.
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize the pixel values to the range [0, 1]
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# One-Hot Encode the labels (10 classes)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# --- 3. Compile and Train the Model ---
print("\n--- Compiling and Training the Model (5 Epochs) ---")

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, 
                    epochs=5, 
                    batch_size=64, 
                    # Use test data as validation data for direct comparison
                    validation_data=(test_images, test_labels),
                    verbose=1)

# --- 4. Evaluate the Model ---
print("\n--- Final Evaluation on Test Data ---")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

# Print the final result clearly
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Compare with the Dense Network from Chapter 2 (97.8%)
print("\n--- Performance Comparison ---")
print(f"Dense Network Accuracy (Ch 2): 0.9780")
print(f"ConvNet Accuracy (Ch 5): {test_acc:.4f}")
if test_acc > 0.978:
    print("Conclusion: The ConvNet significantly outperforms the Dense network.")
else:
    print("Conclusion: The performance is comparable or slightly lower.")
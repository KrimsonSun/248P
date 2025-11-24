# ----------------------------------------------------
# IMPORTS (Adjusted for TensorFlow 2.x compatibility)
# --- FIX: Using 'tensorflow.keras' to prevent import conflicts.
# ----------------------------------------------------
import os

# We no longer import 'keras' separately to avoid the conflict.
# We get Keras and its layers directly from the TensorFlow namespace.
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# We still need numpy
import numpy as np
import time

# ----------------------------------------------------
# SECTION 2.1: A first look at a neural network
# ----------------------------------------------------

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Check data attributes
print("--- 2.1.1 Original Data Check ---")
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
print(test_labels)

# Define the model architecture
model = keras.Sequential(
    [
        # layers is now explicitly 'tensorflow.keras.layers'
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],)

# Data Preprocessing: Reshape and Normalize
print("--- 2.1.2 Data Preprocessing ---")
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# Train the model (Fit)
print("--- 2.1.3 Model Training (Fit) ---")
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Prediction and Evaluation
print("--- 2.1.4 Prediction and Evaluation ---")
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(predictions[0]) # Predicted probability distribution
print(predictions[0].argmax()) # Predicted class
print(predictions[0][7]) # Probability for class 7
print(test_labels[0]) # True label

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")

# ----------------------------------------------------
# SECTION 2.2: Data representations for neural networks
# ----------------------------------------------------

# Scalars (Rank-0 Tensors)
print("--- 2.2.1 Scalars ---")
x = np.array(12)
print(x)
print(x.ndim)

# Vectors (Rank-1 Tensors)
print("--- 2.2.2 Vectors ---")
x = np.array([12, 3, 6, 14, 7])
print(x)
print(x.ndim)

# Matrices (Rank-2 Tensors)
print("--- 2.2.3 Matrices ---")
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x.ndim)

# Rank-3 tensors and higher-rank tensors
print("--- 2.2.4 Rank-3 ---")
x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])
print(x.ndim)

# Key attributes
print("--- 2.2.5 Key Attributes ---")
# Reload original data (as it was reshaped earlier)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

# Manipulating tensors in NumPy (Slicing)
print("--- 2.2.6 Tensor Slicing ---")
my_slice = train_images[10:100]
print(my_slice.shape)
my_slice = train_images[10:100, :, :]
print(my_slice.shape)
my_slice = train_images[10:100, 0:28, 0:28]
print(my_slice.shape)
my_slice = train_images[:, 14:, 14:]
my_slice = train_images[:, 7:-7, 7:-7]

# The notion of data batches
print("--- 2.2.7 Data Batches ---")
batch = train_images[:128]
batch = train_images[128:256]
n = 3
batch = train_images[128 * n : 128 * (n + 1)]


# ----------------------------------------------------
# SECTION 2.3: The gears of neural networks: Tensor operations
# ----------------------------------------------------

# Element-wise operations (Naive Implementations)
print("--- 2.3.1 Element-wise Operations (Naive Implementations) ---")
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x
import time

x = np.random.random((20, 100))
y = np.random.random((20, 100))

# NumPy operation timing
t0 = time.time()
for _ in range(1000):
    z = x + y
    z = np.maximum(z, 0.0)
print("Took (NumPy): {0:.2f} s".format(time.time() - t0))

# Naive operation timing
t0 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(z)
print("Took (Naive): {0:.2f} s".format(time.time() - t0))


# Broadcasting
print("--- 2.3.2 Broadcasting ---")
X = np.random.random((32, 10))
y = np.random.random((10,))
y = np.expand_dims(y, axis=0) # [10,] -> [1, 10]
Y = np.tile(y, (32, 1)) # [1, 10] -> [32, 10]
# The actual broadcasting occurs when running X + y
z = X + y 
print("X + y shape (Broadcasting result):", z.shape)


# Tensor product (Dot product)
print("--- 2.3.3 Tensor Product (Dot Product) ---")
x = np.random.random((32,))
y = np.random.random((32,))
z = np.matmul(x, y)
print("Vector Dot Product (matmul):", z)
z = x @ y
print("Vector Dot Product (@ operator):", z)

# ----------------------------------------------------
# END OF REQUIRED SECTIONS (2.1 - 2.3)
# ----------------------------------------------------
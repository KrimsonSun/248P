import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

# --- 1. Load the Dataset ---
# Load the Reuters dataset, restricting to the top 10,000 most frequent words.
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)

# The Reuters dataset has 46 classes (topics)
NUM_CLASSES = 46

# --- 2. Data Preparation/Encoding ---

def vectorize_sequences(sequences, dimension=10000):
    """
    Converts integer sequences (word indices) into a multi-hot encoded matrix.
    Each review becomes a vector of size 10000 with 1s at the indices corresponding
    to the words present in the sequence.
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # Set specific indices for the words in the sequence to 1.0
        results[i, sequence] = 1.
    return results

# Encode the features (input data)
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Encode the labels using One-Hot Encoding (necessary for categorical_crossentropy loss)
# We have 46 categories, so each label is a vector of size 46.
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)


# --- 3. Model Definition (Standard Configuration C1) ---

def get_standard_model():
    """
    Defines the standard model structure for this multiclass task:
    Two intermediate layers with 64 units, and a final 46-unit softmax layer.
    """
    
    '''
    model = keras.Sequential([
        # First intermediate layer (64 units, ReLU activation)
        layers.Dense(64, activation="relu"),
        # Second intermediate layer (64 units, ReLU activation)
        #layers.Dense(64, activation="relu"),
        # Turn second intermediate layer into 4 units(still ReLU activation), 
        #to test information bottleneck
        #layers.Dense(4, activation="relu"),
        # Output layer (46 units, Softmax activation for multiclass classification)
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    '''
    #smaller model to test the result with smaller capacity.
    model = keras.Sequential([
        # First intermediate layer (32 units, ReLU activation)
        layers.Dense(32, activation="relu"),
        # Second intermediate layer (32 units, ReLU activation)
        layers.Dense(32, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    
    # Compile the model
    model.compile(optimizer="rmsprop",
                  # Standard loss for multiclass classification with probability output
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# --- 4. Training the Final Model ---
# The notebook analysis identified 9 epochs as optimal to avoid severe overfitting.

model = get_standard_model()

print("--- Training Final Model (2x64 units, 9 epochs) ---")

# Train the model using the full training dataset
history = model.fit(
    x_train,
    y_train,
    epochs=9,  # Optimal number of epochs from the notebook
    batch_size=512,
    verbose=1
)

# --- 5. Evaluation ---
print("\n--- Evaluating on Test Data ---")
# Evaluate the model on the unseen test data
results = model.evaluate(x_test, y_test, verbose=0)

print("\n--- Final Results for Standard Configuration ---")
print(f"Test Loss (Categorical Crossentropy): {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")

# Optional: Generate predictions on new data
# predictions = model.predict(x_test)
# print(f"\nFirst prediction vector sums to: {np.sum(predictions[0]):.4f}") # Should be close to 1.0
# print(f"First prediction class index: {np.argmax(predictions[0])}")
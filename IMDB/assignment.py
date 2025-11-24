import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

# --- 1. Load the dataset ---
# Load the IMDb dataset, keeping only the top 10,000 most frequent words.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

# --- 2. Data Preparation/Encoding ---
# Utility function to convert integer sequences to multi-hot encoded vectors.
def vectorize_sequences(sequences, dimension=10000):
    """
    Creates an all-zero matrix of shape (len(sequences), dimension) 
    and sets results[i, j] = 1. for words present in the sequence.
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

# Encode the training and test data.
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Convert labels to float32 type for the model.
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# --- 3. Define the Model (Best Configuration) ---
def get_best_model():
    """
    Defines the optimal model structure found in the notebook:
    2 intermediate layers with 16 units and a final sigmoid layer.
    """
    model = keras.Sequential([
        # First intermediate layer (16 units, ReLU activation)
        layers.Dense(16, activation="relu"),
        # Second intermediate layer (16 units, ReLU activation)
        layers.Dense(16, activation="relu"),
        # Third intermediate layer(same), for alternative test
        #layers.Dense(16, activation="relu"),


        # Output layer (1 unit, Sigmoid activation for binary classification)
        layers.Dense(1, activation="sigmoid")
    ])
    
    # Compile the model using 'rmsprop' (standard for this example in the book)
    model.compile(optimizer="rmsprop",
                  # Standard loss for binary classification with probability output
                  #loss="binary_crossentropy",
                  loss="mse",
                  metrics=["accuracy"])
    return model

# --- 4. Training the Final Model ---
# The notebook identified 4 epochs as optimal to avoid severe overfitting.
model = get_best_model()

print("Training final model for 4 epochs...")
model.fit(
    x_train,
    y_train,
    epochs=4, # Optimal number of epochs
    batch_size=512,
    verbose=1 # Set to 1 to see progress
)

# --- 5. Evaluation ---
print("\nEvaluating on test data...")
results = model.evaluate(x_test, y_test, verbose=0)

print("\n--- Final Results ---")
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")

# End of code for the best configuration
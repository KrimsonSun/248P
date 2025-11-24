import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import boston_housing

# --- 1. Load the Dataset ---
# Load the Boston Housing Price dataset.
# The targets (prices) are in thousands of dollars.
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(f"Training Data Shape: {train_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

# --- 2. Data Normalization (Crucial for Regression) ---
# Normalize the features (input data) to have zero mean and unit variance.
# This is done by subtracting the mean and dividing by the standard deviation.
# We MUST use the statistics (mean and std) derived ONLY from the training data.

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

# Apply the SAME normalization (using train data's stats) to the test data.
test_data -= mean
test_data /= std


# --- 3. Model Definition ---
def build_model():
    """
    Defines a simple, small model (2 hidden layers) suitable for small datasets.
    The output layer has 1 unit with no activation (linear activation) for regression.
    """

    '''
    #Standard model 
    model = keras.Sequential([
        # Intermediate Layer 1: 64 units, ReLU
        layers.Dense(64, activation="relu"),
        # Intermediate Layer 2: 64 units, ReLU
        layers.Dense(64, activation="relu"),
        # Output Layer: 1 unit, no activation (linear) for predicting a continuous value
        layers.Dense(1)
    ])
    '''
    
    '''
    #Small Capacity
    model = keras.Sequential([
        # Intermediate Layer 1: 32 units, ReLU
        layers.Dense(32, activation="relu"),
        # Intermediate Layer 2: 32 units, ReLU
        layers.Dense(32, activation="relu"),
        # Output Layer: 1 unit, no activation (linear) for predicting a continuous value
        layers.Dense(1)
    ])
    '''
    #Small Capacity
    model = keras.Sequential([
        # Intermediate Layer 1: 64 units, ReLU
        layers.Dense(64, activation="relu"),
        # Intermediate Layer 2: 64 units, ReLU
        layers.Dense(64, activation="relu"),
        # Intermediate Layer 3: 64 units, ReLU
        layers.Dense(64, activation="relu"),
        # Output Layer: 1 unit, no activation (linear) for predicting a continuous value
        layers.Dense(1)
    ])
    


    # Compile the model
    # Loss: 'mse' (Mean Squared Error) is the standard for regression.
    # Metric: 'mae' (Mean Absolute Error) is easier to interpret (average error in $1000s).
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


# --- 4. K-Fold Validation (k=4) ---
# Use K-fold validation to get a stable estimate of the model's performance on this small dataset.

k = 4
num_val_samples = len(train_data) // k
#num_epochs = 100

#Alternative ： smaller num

num_epochs =50

all_scores = []

print("\n--- Starting K-fold Validation ---")

for i in range(k):
    print(f"Processing fold #{i+1}")
    
    # Prepare validation data: data from partition i
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    # Prepare training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    
    # Build the Keras model (compiled from scratch for each fold)
    model = build_model()
    
    # Train the model (verbose=0 to suppress output)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=16, verbose=0)
    
    # Evaluate the model on the validation data
    _, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print("\n--- K-Fold Validation Results ---")
print(f"MAE scores per fold: {all_scores}")
print(f"Mean MAE: {np.mean(all_scores):.2f}")
print(f"Standard Deviation: {np.std(all_scores):.2f}")
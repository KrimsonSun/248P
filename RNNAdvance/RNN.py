# advanced_rnn_jena_climate.py
# Objective: Implement two advanced RNN models (Recurrent Dropout GRU and Bidirectional GRU)
#            for the Jena climate time series forecasting problem.

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

# --- Configuration (MUST BE MODIFIED by the user) ---
# NOTE: Please change this path to where you stored the 'jena_climate_2009_2016.csv' file.
DATA_DIR = './data/' 
FNAME = os.path.join(DATA_DIR, 'jena_climate_2009_2016.csv')

# --- Time Series Parameters ---
LOOKBACK = 1440  # Observations go back 10 days (1440 timesteps)
STEP = 6         # Sample at one data point per hour (every 6 timesteps)
DELAY = 144      # Target is 24 hours (144 timesteps) in the future
BATCH_SIZE = 128
TRAIN_SAMPLES = 200000  # First 200k timesteps for training
VAL_SAMPLES_MAX = 300000 # Next 100k timesteps for validation
TEST_SAMPLES_MIN = 300001 # Remainder for testing

# --- Model Parameters ---
GRU_UNITS = 32
DROPOUT_RATE = 0.2
RECURRENT_DROPOUT_RATE = 0.2
EPOCHS = 5 # Increased epochs for models with dropout

# --- 1. Data Loading and Normalization ---

# Load raw data
try:
    with open(FNAME, encoding='utf-8') as f:
        data = f.read()
except FileNotFoundError:
    print(f"ERROR: Data file not found at {FNAME}. Please update the DATA_DIR variable.")
    exit()

lines = data.split('\n')
header = lines[0].split(',')[1:] # Skip 'Date Time'
lines = lines[1:]

print(f"Found {len(lines)} data points.")
print(f"Features: {header}")

# Convert to Numpy array
float_data = np.zeros((len(lines), len(header)))
for i, line in enumerate(lines):
    # Skip the timestamp (index 0)
    values = [float(x.strip('"')) for x in line.split(',')[1:]]
    float_data[i, :] = values

# Normalize the data (using mean and std dev of the training set)
mean = float_data[:TRAIN_SAMPLES].mean(axis=0)
float_data -= mean
std = float_data[:TRAIN_SAMPLES].std(axis=0)
float_data /= std

print(f"Data shape after normalization: {float_data.shape}")

# --- 2. Data Generator Definition ---

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    """
    Yields batches of samples (input sequences) and targets (future temperature).
    """
    if max_index is None:
        max_index = len(data) - delay - 1
    
    # Start index for drawing samples
    i = min_index + lookback
    
    while 1:
        if shuffle:
            # Randomly select rows for shuffling
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            # Draw rows in chronological order
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        
        for j, row in enumerate(rows):
            # Indices for the input sequence (past data)
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            
            # Target is the temperature (index 1) at time: row[j] + delay
            targets[j] = data[rows[j] + delay][1] 
            
        yield samples, targets

# --- 3. Instantiate Generators and Steps ---
train_gen = generator(float_data,
                      lookback=LOOKBACK,
                      delay=DELAY,
                      min_index=0,
                      max_index=TRAIN_SAMPLES,
                      shuffle=True,
                      step=STEP, 
                      batch_size=BATCH_SIZE)

val_gen = generator(float_data,
                    lookback=LOOKBACK,
                    delay=DELAY,
                    min_index=TRAIN_SAMPLES + 1,
                    max_index=VAL_SAMPLES_MAX,
                    step=STEP,
                    batch_size=BATCH_SIZE)

# Calculate steps needed to cover the validation set
val_steps = (VAL_SAMPLES_MAX - (TRAIN_SAMPLES + 1) - LOOKBACK) // BATCH_SIZE

# --- 4. Plotting Function ---
def plot_history(history, model_name):
    """Plots the training and validation MAE loss."""
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, 'bo', label='Training MAE')
    plt.plot(epochs, val_loss, 'b', label='Validation MAE')
    plt.title(f'Training and validation loss ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_loss.png')
    print(f"\nPerformance charts for {model_name} saved to {model_name.lower().replace(' ', '_')}_loss.png")


# --- 5. Model 1: GRU with Recurrent Dropout ---
print("\n" + "="*70)
print(f"Model 1: Single GRU Layer with Dropout (Epochs: {EPOCHS})")

model_dropout = Sequential()
model_dropout.add(layers.GRU(GRU_UNITS,
                             dropout=DROPOUT_RATE,
                             recurrent_dropout=RECURRENT_DROPOUT_RATE,
                             input_shape=(None, float_data.shape[-1])))
model_dropout.add(layers.Dense(1))

model_dropout.compile(optimizer=RMSprop(), loss='mae')

print("\n--- Model 1 Summary (GRU + Recurrent Dropout) ---")
model_dropout.summary()

history_dropout = model_dropout.fit(
    train_gen,
    steps_per_epoch=500,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=val_steps
)

plot_history(history_dropout, "GRU with Recurrent Dropout")


# --- 6. Model 2: Bidirectional GRU ---
print("\n" + "="*70)
print(f"Model 2: Bidirectional GRU Layer (Epochs: {EPOCHS // 2})") 

# Bidirectional models have higher capacity and often converge faster, 
# but we'll use a standard epoch count for comparison.
BIDIR_EPOCHS = EPOCHS // 2 # Use fewer epochs since Bidirectional layers train more parameters

model_bidir = Sequential()
# Bidirectional wrapper duplicates the GRU layer and runs one forward and one backward,
# concatenating their outputs by default.
model_bidir.add(layers.Bidirectional(
    layers.GRU(GRU_UNITS),
    input_shape=(None, float_data.shape[-1])
))
model_bidir.add(layers.Dense(1))

model_bidir.compile(optimizer=RMSprop(), loss='mae')

print("\n--- Model 2 Summary (Bidirectional GRU) ---")
model_bidir.summary()

history_bidir = model_bidir.fit(
    train_gen,
    steps_per_epoch=500,
    epochs=BIDIR_EPOCHS,
    validation_data=val_gen,
    validation_steps=val_steps
)

plot_history(history_bidir, "Bidirectional GRU")

print("\nAssignment complete: Two advanced RNN models have been implemented and trained.")
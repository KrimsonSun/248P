# simple_rnn_lstm_imdb_classifiers.py
# Objective: Implement two RNN classifiers for the IMDB sentiment classification task:
# 1. SimpleRNN Model
# 2. LSTM Model
# This script aligns with the methodology described in Section 6.2 of the textbook.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import History

# --- 1. Configuration Parameters ---
# Use smaller parameters for faster training and demonstration, sufficient to show differences
MAX_FEATURES = 10000  # Vocabulary size (top 10,000 most frequent words)
MAX_LEN = 100         # Truncate/pad sequences to 100 words
EMBEDDING_DIM = 32    # Dimensionality of the embedding vector
RNN_UNITS = 32        # Output dimensionality of the RNN layer
BATCH_SIZE = 128
EPOCHS = 10           # Increased epochs to observe convergence
LEARNING_RATE = 1e-4

print("--- Configuration Loaded ---")
print(f"MAX_FEATURES: {MAX_FEATURES}, MAX_LEN: {MAX_LEN}, RNN_UNITS: {RNN_UNITS}")

# --- 2. Data Loading and Preprocessing ---
print('\nLoading IMDB data...')
# Load data, limiting to MAX_FEATURES (top 10000 words)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

print(f'{len(x_train)} train sequences')
print(f'{len(x_test)} test sequences')

print(f'Pad sequences (samples x time) to maxlen={MAX_LEN}...')
# Pad the sequences to the consistent length (MAX_LEN)
# 'post' padding is common for RNNs, though 'pre' is often default.
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# --- Plotting Utility ---
def plot_history(history: History, model_name: str):
    """Plots the training and validation accuracy and loss."""
    # Keras History object stores metrics
    acc = history.history.get('acc') or history.history.get('accuracy')
    val_acc = history.history.get('val_acc') or history.history.get('val_accuracy')
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(f'Training and validation accuracy ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(f'Training and validation loss ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_performance.png')
    plt.close() # Close plot to prevent display issues in some environments
    print(f"\nPerformance charts for {model_name} saved to {model_name.lower().replace(' ', '_')}_performance.png")


# ======================================================================
# Classifier 1: SimpleRNN Model
# Structure: Embedding -> SimpleRNN -> Dense
# ======================================================================
print("\n" + "="*70)
print("Building and training SimpleRNN Classifier...")

simplernn_model = Sequential()
# 1. Embedding layer: Maps words (indices) to dense vectors
simplernn_model.add(Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN)) 

# 2. SimpleRNN layer: Processes the sequence. Output shape is (batch_size, RNN_UNITS)
# Note: SimpleRNN struggles with long sequences due to vanishing gradient.
simplernn_model.add(SimpleRNN(RNN_UNITS)) 

# 3. Output Dense layer: Sigmoid activation for binary classification (sentiment)
simplernn_model.add(Dense(1, activation='sigmoid'))

simplernn_model.compile(optimizer=RMSprop(learning_rate=LEARNING_RATE), 
                        loss='binary_crossentropy', 
                        metrics=['acc'])

print("\n--- SimpleRNN Model Summary ---")
simplernn_model.summary()

simplernn_history = simplernn_model.fit(
    x_train, 
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2, # Use 20% of training data for validation
    verbose=1
)

plot_history(simplernn_history, "SimpleRNN Classifier")


# ======================================================================
# Classifier 2: LSTM Model
# Structure: Embedding -> LSTM -> Dense
# ======================================================================
print("\n" + "="*70)
print("Building and training LSTM Classifier...")

lstm_model = Sequential()
# 1. Embedding layer
lstm_model.add(Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN)) 

# 2. LSTM layer: Uses gates (Forget, Input, Output) to better manage sequence state, 
# overcoming the vanishing gradient problem inherent in SimpleRNN.
lstm_model.add(LSTM(RNN_UNITS)) 

# 3. Output Dense layer
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer=RMSprop(learning_rate=LEARNING_RATE), 
                   loss='binary_crossentropy', 
                   metrics=['acc'])

print("\n--- LSTM Model Summary ---")
lstm_model.summary()

lstm_history = lstm_model.fit(
    x_train, 
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2, # Use 20% of training data for validation
    verbose=1
)

plot_history(lstm_history, "LSTM Classifier")

print("\n--- Execution Complete ---")
print("Two RNN-based classifiers (SimpleRNN and LSTM) have been trained and results plotted.")
print("By comparing the validation accuracy charts of SimpleRNN and LSTM, you should observe that LSTM generally performs better in handling temporal dependencies.")
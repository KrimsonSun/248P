# rnn_classifiers.py
# Objective: Implement SimpleRNN and LSTM classifiers for the IMDB dataset.

import numpy as np
import matplotlib.pyplot as plt
# Use modern TensorFlow Keras paths for all imports
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

# --- 1. Configuration Parameters (Based on Textbook Excerpt) ---
MAX_FEATURES = 10000  # Number of words to consider as features
MAXLEN = 500          # Cut texts after this number of words
EMBEDDING_DIM = 32    # Dimensionality of the embedding vector
RNN_UNITS = 32        # Number of units in the SimpleRNN/LSTM layer
BATCH_SIZE = 128      # Batch size used for training
EPOCHS = 10

# --- 2. Data Loading and Preprocessing ---
print('Loading data...')
# Load data, limiting to MAX_FEATURES (top 10000 words)
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

print(f'{len(input_train)} train sequences')
print(f'{len(input_test)} test sequences')

print(f'Pad sequences (samples x time) to maxlen={MAXLEN}...')
# Pad the sequences to the consistent length (MAXLEN)
input_train = pad_sequences(input_train, maxlen=MAXLEN)
input_test = pad_sequences(input_test, maxlen=MAXLEN)

print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


def plot_history(history, model_name):
    """Plots the training and validation accuracy and loss."""
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(f'Training and validation accuracy ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(f'Training and validation loss ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_performance.png')
    print(f"\nPerformance charts for {model_name} saved to {model_name.lower().replace(' ', '_')}_performance.png")


# ----------------------------------------------------------------------
# Classifier 1: SimpleRNN Model
# ----------------------------------------------------------------------
print("\n" + "="*50)
print("Building and training SimpleRNN classifier...")

simple_rnn_model = Sequential()
simple_rnn_model.add(Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAXLEN))
# SimpleRNN layer returns only the last output (2D tensor) by default
simple_rnn_model.add(SimpleRNN(RNN_UNITS))
simple_rnn_model.add(Dense(1, activation='sigmoid'))

simple_rnn_model.compile(optimizer='rmsprop', 
                         loss='binary_crossentropy', 
                         metrics=['acc'])

print("\n--- SimpleRNN Model Summary ---")
simple_rnn_model.summary()

simple_rnn_history = simple_rnn_model.fit(
    input_train, 
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2, # Validation split as per the textbook example
    verbose=1
)

plot_history(simple_rnn_history, "SimpleRNN")

# ----------------------------------------------------------------------
# Classifier 2: LSTM Model
# ----------------------------------------------------------------------
print("\n" + "="*50)
print("Building and training LSTM classifier...")

lstm_model = Sequential()
lstm_model.add(Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAXLEN))
# LSTM layer (often performs much better than SimpleRNN on long sequences)
lstm_model.add(LSTM(RNN_UNITS))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer='rmsprop', 
                   loss='binary_crossentropy', 
                   metrics=['acc'])

print("\n--- LSTM Model Summary ---")
lstm_model.summary()

lstm_history = lstm_model.fit(
    input_train, 
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2, # Validation split as per the textbook example
    verbose=1
)

plot_history(lstm_history, "LSTM")

print("\nAssignment complete. Two models were trained and performance charts were saved.")
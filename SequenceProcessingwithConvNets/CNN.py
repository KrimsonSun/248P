# convnet_sequence_classifiers.py
# Objective: Implement two 1D ConvNet classifiers for the IMDB sentiment classification task:
# 1. A pure 1D ConvNet model.
# 2. A hybrid model combining 1D ConvNets for feature extraction and a GRU layer for sequence modeling.

import numpy as np
import matplotlib.pyplot as plt
# Use modern TensorFlow Keras paths for all imports
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GRU, Dense
from tensorflow.keras.optimizers import RMSprop

# --- 1. Configuration Parameters ---
MAX_FEATURES = 10000  # Number of words to consider as features (top 10k)
MAX_LEN = 500         # Cut texts after this number of words
EMBEDDING_DIM = 128   # Dimensionality of the embedding vector (as used in the text example)
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-4

# --- 2. Data Loading and Preprocessing ---
print('Loading IMDB data...')
# Load data, limiting to MAX_FEATURES (top 10000 words)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

print(f'{len(x_train)} train sequences')
print(f'{len(x_test)} test sequences')

print(f'Pad sequences (samples x time) to maxlen={MAX_LEN}...')
# Pad the sequences to the consistent length (MAX_LEN)
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# --- Plotting Utility ---
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
# Classifier 1: Simple 1D ConvNet Model
# Structure: Embedding -> Conv1D -> MaxPooling1D -> Conv1D -> GlobalMaxPooling1D -> Dense
# ----------------------------------------------------------------------
print("\n" + "="*70)
print("Building and training Simple 1D ConvNet Classifier...")

convnet_model = Sequential()
# 1. Embedding layer (Input shape: (None, 500))
convnet_model.add(Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN)) 
# 2. First 1D Convolution (Window size 7)
convnet_model.add(Conv1D(32, 7, activation='relu'))
# 3. Downsampling (Pool size 5)
convnet_model.add(MaxPooling1D(5))
# 4. Second 1D Convolution (Window size 7)
convnet_model.add(Conv1D(32, 7, activation='relu'))
# 5. Global Pooling to flatten time dimension (Output shape: (None, 32))
convnet_model.add(GlobalMaxPooling1D())
# 6. Output Dense layer
convnet_model.add(Dense(1, activation='sigmoid'))

convnet_model.compile(optimizer=RMSprop(learning_rate=LEARNING_RATE), 
                      loss='binary_crossentropy', 
                      metrics=['acc'])

print("\n--- Simple 1D ConvNet Model Summary ---")
convnet_model.summary()

convnet_history = convnet_model.fit(
    x_train, 
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

plot_history(convnet_history, "Simple ConvNet")


# ----------------------------------------------------------------------
# Classifier 2: CNN-RNN Hybrid Model
# Structure: Embedding -> Conv1D -> MaxPooling1D -> GRU -> Dense
# Objective: Use CNN for feature extraction/downsampling, then GRU for order-sensitivity.
# ----------------------------------------------------------------------
print("\n" + "="*70)
print("Building and training CNN-RNN Hybrid Classifier...")

hybrid_model = Sequential()
# 1. Embedding layer
hybrid_model.add(Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN)) 
# 2. 1D Convolution for feature extraction (Window size 5, as often used in hybrid approaches)
hybrid_model.add(Conv1D(32, 5, activation='relu'))
# 3. Downsampling the sequence (Pool size 5)
hybrid_model.add(MaxPooling1D(5))
# 4. GRU layer for sequence processing (Note: recurrent_dropout is omitted for CuDNN speedup)
hybrid_model.add(GRU(32, dropout=0.1)) 
# 5. Output Dense layer
hybrid_model.add(Dense(1, activation='sigmoid'))

# Using a slightly higher learning rate than the pure ConvNet for comparison
hybrid_model.compile(optimizer=RMSprop(), 
                     loss='binary_crossentropy', 
                     metrics=['acc'])

print("\n--- CNN-RNN Hybrid Model Summary ---")
hybrid_model.summary()

hybrid_history = hybrid_model.fit(
    x_train, 
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

plot_history(hybrid_history, "CNN-GRU Hybrid")

print("\n--- Execution Complete ---")
print("Two classifiers (Simple ConvNet and CNN-GRU Hybrid) have been trained and results plotted.")
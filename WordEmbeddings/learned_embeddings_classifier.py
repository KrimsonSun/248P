# learned_embeddings_classifier.py

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
# The following modules are corrected to use the functional path:
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# --- 1. Data Preparation Parameters ---
# Number of words to consider as features (top 10000 most common words)
MAX_FEATURES = 10000
# Cut texts after this number of words
MAXLEN = 20
# Dimensionality of the learned embedding vectors
EMBEDDING_DIM = 8

print("Loading IMDB data...")
# Load the data as lists of integers, restricting to MAX_FEATURES words.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

# This turns our lists of integers into a 2D integer tensor of shape `(samples, maxlen)`
print(f"Padding sequences to maxlen={MAXLEN}...")
# CRITICAL FIX: Calling pad_sequences directly, as it was imported directly.
x_train = pad_sequences(x_train, maxlen=MAXLEN)
x_test = pad_sequences(x_test, maxlen=MAXLEN)

# --- 2. Model Definition ---
model = Sequential()
# Embedding layer: learns an 8-dimensional vector for each of the 10,000 words.
model.add(Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAXLEN))

# Flatten layer: flattens the 3D tensor (samples, maxlen, 8) 
# into a 2D tensor (samples, maxlen * 8) for the Dense layer.
model.add(Flatten())

# Classifier layer: a single Dense layer for binary classification
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

print("\n--- Model Summary (Learned Embeddings) ---")
model.summary()

# --- 3. Model Training ---
print("\nTraining model...")
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

# --- 4. Plotting Results ---
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy (Learned Embeddings)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss (Learned Embeddings)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plots as the execution environment might not display them
plt.savefig('learned_embeddings_performance.png')
print("\nPerformance charts saved to learned_embeddings_performance.png")
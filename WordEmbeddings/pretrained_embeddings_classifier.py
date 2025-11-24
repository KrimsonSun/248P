# pretrained_embeddings_classifier.py

import os
import numpy as np
import matplotlib.pyplot as plt
# New import paths (resolves error in TF 2.x)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# --- Configuration (MUST BE MODIFIED by the user) ---
# NOTE: You MUST change these paths to where you saved the IMDB and GloVe files.
IMDB_DIR = './aclImdb'  # Path to the uncompressed IMDB raw text data
GLOVE_DIR = './glove_data/' # Path to the directory containing glove.6B.100d.txt

MAXLEN = 100       # We will cut reviews after 100 words
TRAINING_SAMPLES = 200  # We will be training on 200 samples only (as per assignment)
VALIDATION_SAMPLES = 10000  # Validation set size
MAX_WORDS = 10000  # Only consider the top 10,000 words in the dataset
EMBEDDING_DIM = 100 # GloVe 6B is 100-dimensional

# --- 1. Data Preparation: Loading Raw Text ---
print("Loading raw IMDB text data...")
train_dir = os.path.join(IMDB_DIR, 'train')
labels = []
texts = []

# Collect positive and negative reviews
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    if not os.path.isdir(dir_name):
        print(f"Error: Directory not found. Check IMDB_DIR path: {dir_name}")
        exit()
        
    for fname in os.listdir(dir_name):
        if fname.endswith('.txt'):
            # Using 'with open' is safer than f.open()
            with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                texts.append(f.read())
            labels.append(0 if label_type == 'neg' else 1)

# --- 2. Text Vectorization and Dataset Splitting ---
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print(f'Found {len(word_index)} unique tokens.')

data = pad_sequences(sequences, maxlen=MAXLEN)
labels = np.asarray(labels)

# Shuffle the data since it was collected in order (neg then pos)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Split the small training set and the validation set
x_train = data[:TRAINING_SAMPLES]
y_train = labels[:TRAINING_SAMPLES]
x_val = data[TRAINING_SAMPLES: TRAINING_SAMPLES + VALIDATION_SAMPLES]
y_val = labels[TRAINING_SAMPLES: TRAINING_SAMPLES + VALIDATION_SAMPLES]

print(f"Shape of training data (200 samples): {x_train.shape}")

# --- 3. Processing GloVe Embeddings File ---
print("\nProcessing GloVe embeddings...")
embeddings_index = {}
glove_file_path = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')

try:
    with open(glove_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f'Found {len(embeddings_index)} word vectors.')
except FileNotFoundError:
    print(f"ERROR: GloVe file not found at {glove_file_path}. Please check GLOVE_DIR path.")
    exit()

# Build the Embedding Matrix
embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Load GloVe vector if found; otherwise, it remains all-zeros.
            embedding_matrix[i] = embedding_vector

# --- 4. Model Definition (FIXED) ---
model = Sequential()
# FIX: Pass the embedding_matrix and trainable=False directly during layer construction.
model.add(Embedding(
    MAX_WORDS, 
    EMBEDDING_DIM, 
    weights=[embedding_matrix],  # Pass pre-trained weights here
    input_length=MAXLEN,
    trainable=False              # Freeze weights here
))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# --- 5. Load and Freeze GloVe Weights (Manual steps removed) ---
# The weights are now loaded and frozen in the model definition above.
# We skip the failing set_weights call.

print("\n--- Model Summary (Pre-trained Embeddings) ---")
model.summary()

# --- 6. Model Training ---
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

print("\nTraining model with frozen GloVe embeddings...")
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Save the trained weights for later evaluation on the test set
# FIX: Changed file extension from .h5 to .weights.h5 as required by Keras/TF version.
model.save_weights('pre_trained_glove_model.weights.h5')

# --- 7. Plotting Results ---
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy (GloVe)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss (GloVe)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('pretrained_embeddings_performance.png')
print("\nPerformance charts saved to pretrained_embeddings_performance.png")
import os
import sys
import shutil
import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# --- 1. Command Line Argument and Path Setup ---

# Check the number of command-line arguments
if len(sys.argv) != 3:
    print("Usage: python cats_dogs_small.py <original_image_dir> <new_base_dir>")
    print("Example: python cats_dogs_small.py /path/to/dogs-vs-cats /path/to/smaller")
    sys.exit(1)

# Get command-line arguments
original_dir_path = sys.argv[1]
new_base_dir_path = sys.argv[2]

# Convert paths to pathlib objects for easier manipulation
original_image_dir = pathlib.Path(original_dir_path)
new_base_dir = pathlib.Path(new_base_dir_path)

# Define subset sizes
train_size = 1000  # 1000 cat images + 1000 dog images
validation_size = 500  # 500 cat + 500 dog
test_size = 1000  # 1000 cat + 1000 dog

# Ensure the target directory is clean and create the new directory structure
if new_base_dir.exists():
    shutil.rmtree(new_base_dir) # Delete old directory to ensure a clean run
new_base_dir.mkdir(parents=True, exist_ok=True)


# --- 2. Data Subsetting Function ---

def make_subset(subset_name, start_index, end_index):
    """
    Copies images from the original directory to the new subset directories.
    """
    print(f"Creating {subset_name} subset ({end_index - start_index} images per category)...")
    for category in ("cat", "dog"):
        # Construct the target subdirectory path (e.g., cats_vs_dogs_small/train/cat)
        dir_to_create = new_base_dir / subset_name / category
        os.makedirs(dir_to_create, exist_ok=True)
        
        # Construct the list of filenames to copy (e.g., cat.0.jpg, cat.1.jpg, ...)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        
        for fname in fnames:
            src = original_image_dir / fname
            dst = dir_to_create / fname
            
            # Check if the source file exists to prevent crashing due to incorrect paths
            if src.exists():
                shutil.copyfile(src, dst)
            else:
                print(f"Warning: Source file not found: {src}. Check your original data path.")


# Execute data splitting
make_subset("train", start_index=0, end_index=train_size)
make_subset("validation", start_index=train_size, end_index=train_size + validation_size)
make_subset("test", start_index=train_size + validation_size, end_index=train_size + validation_size + test_size)


# --- 3. Model Definition ---

# Input image size 
IMAGE_SIZE = (180, 180)
INPUT_SHAPE = IMAGE_SIZE + (3,) # (180, 180, 3)

# Model Definition (Data augmentation is handled by ImageDataGenerator for TF 2.3.0 compatibility)
inputs = keras.Input(shape=INPUT_SHAPE)
x = inputs # Input goes directly to the first Conv2D layer

# Convolutional Base (Feature Extractor)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)

# Classifier
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x) # Introduce Dropout (0.5) as a regularization technique
outputs = layers.Dense(1, activation="sigmoid")(x) # Use sigmoid activation for binary classification

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# --- 4. Model Compilation ---

# Use binary_crossentropy loss for binary classification task
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

# --- 5. Data Loading using ImageDataGenerator (Compatible Approach) ---

BATCH_SIZE = 32

# 1. Instantiate the generator, specify rescaling AND augmentation
# These parameters are compatible with TensorFlow 2.3.0
datagen = ImageDataGenerator(
    rescale=1./255,          # Rescaling
    rotation_range=10,       # Random rotation 10 degrees (corresponds to RandomRotation(0.1) in modern Keras)
    zoom_range=0.2,          # Random zoom 20%
    horizontal_flip=True,    # Horizontal flip
)

# 2. Use flow_from_directory to create the datasets
# Note: Augmentation is only applied to the training generator.
train_generator = datagen.flow_from_directory(
    new_base_dir / "train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary')

# For validation and test sets, we only need rescaling, so we use a simpler generator.
validation_test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_test_datagen.flow_from_directory(
    new_base_dir / "validation",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary')

test_generator = validation_test_datagen.flow_from_directory(
    new_base_dir / "test",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False) # Important: Disable shuffle for predictable test evaluation

# Determine steps per epoch for the generator (size of dataset / batch size)
# train_size * 2 = 2000 images (1000 cat + 1000 dog)
STEPS_PER_EPOCH_TRAIN = (train_size * 2) // BATCH_SIZE
# validation_size * 2 = 1000 images
STEPS_PER_EPOCH_VAL = (validation_size * 2) // BATCH_SIZE
# test_size * 2 = 2000 images
STEPS_PER_EPOCH_TEST = (test_size * 2) // BATCH_SIZE


# --- 6. Training with Model Checkpoint ---

# ModelCheckpoint callback: Saves only the best model based on the minimum validation loss
CALLBACKS = [
    keras.callbacks.ModelCheckpoint(
        filepath="cats_dogs_small.keras",
        save_best_only=True,
        monitor="val_loss")]

# Train the model using the generators and steps_per_epoch argument
print("\n--- Starting Model Training (30 Epochs) ---")
history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH_TRAIN, # Required when using generator
    epochs=30,
    validation_data=validation_generator,
    validation_steps=STEPS_PER_EPOCH_VAL,  # Required when using generator
    callbacks=CALLBACKS
)


# --- 7. Plotting Results ---

def plot_history(history):
    """Plots the training and validation accuracy and loss curves."""
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

print("\n--- Displaying Training and Validation Curves ---")
plot_history(history)


# --- 8. Final Evaluation ---

# Load the best saved model
try:
    test_model = keras.models.load_model("cats_dogs_small.keras")
    print("\n--- Evaluating Best Saved Model on Test Set ---")
    # Evaluate the model using the test generator and steps argument
    test_loss, test_acc = test_model.evaluate(test_generator, steps=STEPS_PER_EPOCH_TEST)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
except Exception as e:
    print(f"\nError during final evaluation: {e}")
    print("Could not load the best model (cats_dogs_small.keras). The training might have failed.")
import os
import sys
import shutil
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Keras and TensorFlow imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

# --- Global Configuration ---
# Image dimensions
IMAGE_SIZE = (180, 180)
INPUT_SHAPE = IMAGE_SIZE + (3,)
BATCH_SIZE = 32

# Subset sizes (as defined in the notebook/book for this small dataset)
TRAIN_SIZE = 1000  # 1000 cat + 1000 dog
VALIDATION_SIZE = 500  # 500 cat + 500 dog
TEST_SIZE = 1000  # 1000 cat + 1000 dog
TOTAL_TEST_IMAGES = TEST_SIZE * 2
TOTAL_VALIDATION_IMAGES = VALIDATION_SIZE * 2


## 1. Command Line Argument and Path Setup
# -------------------------------------------------------------
def setup_paths():
    """Parses command line arguments and returns pathlib objects for directories."""
    parser = argparse.ArgumentParser(
        description="Train a Cats vs. Dogs classifier using VGG16 feature extraction and data augmentation."
    )
    parser.add_argument(
        "original_image_dir", 
        type=str, 
        help="Path to the original unzipped 'train' folder (e.g., /home/lopes/Datasets/dogs-vs-cats/train)"
    )
    parser.add_argument(
        "new_base_dir", 
        type=str, 
        help="Path where the smaller, subsetted dataset will be created (e.g., /your/path/to/smaller)"
    )
    args = parser.parse_args()
    
    original_dir_path = pathlib.Path(args.original_image_dir)
    new_base_dir_path = pathlib.Path(args.new_base_dir)

    if not original_dir_path.exists():
        print(f"Error: Original directory not found at {original_dir_path}")
        sys.exit(1)

    # Ensure the target directory is clean and create the new directory structure
    if new_base_dir_path.exists():
        print(f"Removing old base directory: {new_base_dir_path}")
        shutil.rmtree(new_base_dir_path)
    new_base_dir_path.mkdir(parents=True, exist_ok=True)
    
    return original_dir_path, new_base_dir_path

## 2. Data Subsetting Function
# -------------------------------------------------------------
def make_subset(original_image_dir, new_base_dir, subset_name, start_index, end_index):
    """Copies images from the original directory to the new subset directories."""
    print(f"Creating {subset_name} subset...")
    for category in ("cat", "dog"):
        # Construct the target subdirectory path (e.g., new_base_dir/train/cat)
        dir_to_create = new_base_dir / subset_name / category
        os.makedirs(dir_to_create, exist_ok=True)
        
        # Construct the list of filenames to copy
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        
        for fname in fnames:
            src = original_image_dir / fname
            dst = dir_to_create / fname
            
            if src.exists():
                shutil.copyfile(src, dst)
            else:
                print(f"Warning: Source file not found: {src}")

def prepare_data(original_image_dir, new_base_dir):
    """Executes data splitting into train, validation, and test subsets."""
    make_subset(original_image_dir, new_base_dir, "train", start_index=0, end_index=TRAIN_SIZE)
    make_subset(original_image_dir, new_base_dir, "validation", start_index=TRAIN_SIZE, end_index=TRAIN_SIZE + VALIDATION_SIZE)
    make_subset(original_image_dir, new_base_dir, "test", start_index=TRAIN_SIZE + VALIDATION_SIZE, end_index=TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE)

## 3. Model Definition (Feature Extraction with Data Augmentation)
# -------------------------------------------------------------
def build_pretrained_model():
    """
    Instantiates VGG16 and builds the complete model with data augmentation and a classifier head.
    The VGG16 base is frozen.
    """
    print("\n--- Building VGG16-based Model (Feature Extraction) ---")
    
    # Instantiate and freeze the VGG16 convolutional base
    conv_base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    conv_base.trainable = False
    
    # 1. Define Data Augmentation layers
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ],
        name="data_augmentation"
    )

    # 2. Build the Model Pipeline
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = data_augmentation(inputs)
    
    # VGG16 requires its own preprocessing (standardization relative to ImageNet)
    x = keras.applications.vgg16.preprocess_input(x) 
    
    # Pass through the frozen VGG16 base
    x = conv_base(x) 
    
    # 3. Add the Classifier Head
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x) # Binary classification
    
    model = keras.Model(inputs, outputs)
    
    # 4. Compile the model
    # Use a low learning rate for potentially better performance, typical for fine-tuning/transfer learning
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(learning_rate=2e-5), 
        metrics=["accuracy"]
    )
    
    model.summary()
    return model

## 4. Data Loading and Training
# -------------------------------------------------------------
def get_generators(new_base_dir):
    """Creates training and validation generators using ImageDataGenerator."""
    
    # Training Generator (includes Data Augmentation and Rescaling is handled by VGG16.preprocess_input)
    # The rescale is often done *inside* the model for modern Keras APIs, but using ImageDataGenerator
    # here requires minimal setup since VGG16.preprocess_input handles the scaling for us.
    train_datagen = ImageDataGenerator()

    # Validation/Test Generator (only standard preprocessing, no augmentation)
    validation_test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        new_base_dir / "train",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary')

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
        shuffle=False) # Essential for correct test evaluation
        
    return train_generator, validation_generator, test_generator

def train_model(model, train_gen, val_gen):
    """Trains the model and saves the best version."""
    
    # Ensure the model is saved in the required .h5 format
    CALLBACKS = [
        keras.callbacks.ModelCheckpoint(
            filepath="cats_dogs_pretrained.h5", # Required .h5 format
            save_best_only=True,
            monitor="val_loss")
    ]
    
    print("\n--- Starting Model Training (30 Epochs) ---")
    
    # Calculate steps per epoch for the generator approach
    steps_per_epoch_train = (TRAIN_SIZE * 2) // BATCH_SIZE
    steps_per_epoch_val = (VALIDATION_SIZE * 2) // BATCH_SIZE

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch_train,
        epochs=30, # Reduced from 50/100 for faster demonstration/smaller file
        validation_data=val_gen,
        validation_steps=steps_per_epoch_val,
        callbacks=CALLBACKS
    )
    return history

## 5. Plotting Results and Evaluation
# -------------------------------------------------------------
def plot_history(history):
    """Plots the training and validation accuracy and loss curves."""
    print("\n--- Displaying Training and Validation Curves ---")
    
    accuracy = history.history.get("accuracy", history.history.get("acc"))
    val_accuracy = history.history.get("val_accuracy", history.history.get("val_acc"))
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

def evaluate_model(test_gen):
    """Loads the best model and evaluates it on the test set."""
    
    model_filepath = "cats_dogs_pretrained.h5"
    if not os.path.exists(model_filepath):
        print(f"\nError: Best model file not found at {model_filepath}. Training likely failed.")
        return

    try:
        # Steps for test evaluation
        steps_per_epoch_test = TOTAL_TEST_IMAGES // BATCH_SIZE
        
        # Load the best saved model (saved in .h5 format)
        test_model = load_model(model_filepath)
        print("\n--- Evaluating Best Saved Model on Test Set ---")
        
        test_loss, test_acc = test_model.evaluate(test_gen, steps=steps_per_epoch_test)
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")
        
    except Exception as e:
        print(f"\nError during final evaluation: {e}")
        print("Could not load the best model.")


## Main Execution Block
# -------------------------------------------------------------
if __name__ == "__main__":
    try:
        # 1. Setup paths and arguments
        ORIGINAL_DIR, NEW_BASE_DIR = setup_paths()
        
        # 2. Prepare data subsets
        prepare_data(ORIGINAL_DIR, NEW_BASE_DIR)
        
        # 3. Build the model
        model = build_pretrained_model()
        
        # 4. Create data generators
        train_generator, validation_generator, test_generator = get_generators(NEW_BASE_DIR)
        
        # 5. Train the model
        history = train_model(model, train_generator, validation_generator)
        
        # 6. Plot results
        plot_history(history)
        
        # 7. Final evaluation
        evaluate_model(test_generator)

    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")
        sys.exit(1)
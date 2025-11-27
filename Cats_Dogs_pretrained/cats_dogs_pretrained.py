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
from tensorflow.keras.optimizers import RMSprop, Adam # Import Adam for fine-tuning
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
TOTAL_TRAIN_IMAGES = TRAIN_SIZE * 2
TOTAL_TEST_IMAGES = TEST_SIZE * 2
TOTAL_VALIDATION_IMAGES = VALIDATION_SIZE * 2

# Training Epochs Configuration
INITIAL_EPOCHS = 10  # Epochs for Feature Extraction (Stage 1)
FINE_TUNE_EPOCHS = 20 # Epochs for Fine-Tuning (Stage 2)
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# Fine-Tuning Configuration
FINE_TUNE_AT_LAYER = 15 # VGG16 has 19 layers. We'll unfreeze starting from block5_conv1 (layer 15) for partial fine-tuning.


## 1. Command Line Argument and Path Setup
# -------------------------------------------------------------
def setup_paths():
    """Parses command line arguments and returns pathlib objects for directories."""
    parser = argparse.ArgumentParser(
        description="Train a Cats vs. Dogs classifier using VGG16 two-stage transfer learning (Feature Extraction and Fine-Tuning)."
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

## 3. Model Definition and First Compilation (Feature Extraction)
# -------------------------------------------------------------
def build_feature_extractor_model():
    """
    Instantiates VGG16 (frozen) and attaches a custom classifier head.
    This model is compiled for the initial Feature Extraction stage.
    """
    print("\n--- 1. Building VGG16 Feature Extractor Model ---")
    
    # Instantiate VGG16 base (with ImageNet weights, excluding top classifier)
    conv_base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    # Freeze the convolutional base for the Feature Extraction stage
    conv_base.trainable = False
    
    # 1. Define Data Augmentation layers
    # Use modern Keras preprocessing layers which are efficient on GPU/TPU
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
    
    # VGG16 requires ImageNet preprocessing (centering the data)
    x = keras.applications.vgg16.preprocess_input(x) 
    
    # Pass through the frozen VGG16 base
    # Note: training=False is often explicitly passed when using a frozen base inside a Keras model
    x = conv_base(x, training=False) 
    
    # 3. Add the Classifier Head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x) # Use ReLU activation
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x) # Binary classification
    
    model = keras.Model(inputs, outputs)
    
    # 4. Compile the model for Feature Extraction (Stage 1)
    # Use a moderate learning rate (e.g., 1e-3) for the new classification head
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=1e-3), # Using Adam for initial training
        metrics=["accuracy"]
    )
    
    model.summary()
    return model, conv_base # Return both for later fine-tuning


def fine_tune_model(model, conv_base):
    """
    Unfreezes the top layers of the VGG16 base and recompiles the model 
    with a very low learning rate for Fine-Tuning (Stage 2).
    """
    print(f"\n--- 3. Fine-Tuning Stage Setup (Unfreezing Layers from {FINE_TUNE_AT_LAYER}) ---")
    
    # 1. Unfreeze the convolutional base
    conv_base.trainable = True

    # 2. Freeze all layers up to the fine-tuning point
    for layer in conv_base.layers:
        if conv_base.layers.index(layer) < FINE_TUNE_AT_LAYER:
            layer.trainable = False
        else:
            layer.trainable = True

    # 3. Recompile the model with a very low learning rate
    # Crucial step: The entire model must be recompiled for changes to trainable=True to take effect
    model.compile(
        loss="binary_crossentropy",
        # Use a very low learning rate to prevent large weight updates
        optimizer=Adam(learning_rate=1e-5), 
        metrics=["accuracy"]
    )
    
    model.summary()
    return model


## 4. Data Loading and Training
# -------------------------------------------------------------
def get_generators(new_base_dir):
    """
    Creates training, validation, and test generators using ImageDataGenerator.
    Note: Data augmentation layers are defined in the model, so datagen is minimal.
    """
    
    # Generators are only used for loading data from disk, not for augmentation/scaling.
    # The VGG16.preprocess_input handles scaling/centering.
    train_datagen = ImageDataGenerator() # No parameters needed here
    validation_test_datagen = ImageDataGenerator() # No parameters needed here

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

def run_training(model, train_gen, val_gen, stage_name, epochs, initial_epoch=0, filename="model.h5"):
    """Runs a single training stage and saves the best model."""
    
    # Save the best model based on validation loss
    CALLBACKS = [
        keras.callbacks.ModelCheckpoint(
            filepath=filename,
            save_best_only=True,
            monitor="val_loss")
    ]
    
    print(f"\n--- Starting {stage_name} Training ({epochs} Epochs) ---")
    
    # Calculate steps per epoch for the generator approach
    steps_per_epoch_train = TOTAL_TRAIN_IMAGES // BATCH_SIZE
    steps_per_epoch_val = TOTAL_VALIDATION_IMAGES // BATCH_SIZE

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=val_gen,
        validation_steps=steps_per_epoch_val,
        callbacks=CALLBACKS
    )
    return history

## 5. Plotting Results and Evaluation
# -------------------------------------------------------------
def plot_history(history_ft, history_fe, total_epochs, initial_epochs):
    """
    Plots the two-stage training and validation accuracy and loss curves.
    Safely concatenates the two history objects.
    """
    print("\n--- Displaying Training and Validation Curves ---")
    
    # Concatenate histories
    # Use .get() for compatibility with older Keras history keys
    acc_fe = history_fe.history.get("accuracy", history_fe.history.get("acc"))
    val_acc_fe = history_fe.history.get("val_accuracy", history_fe.history.get("val_acc"))
    loss_fe = history_fe.history["loss"]
    val_loss_fe = history_fe.history["val_loss"]

    acc_ft = history_ft.history.get("accuracy", history_ft.history.get("acc"))
    val_acc_ft = history_ft.history.get("val_accuracy", history_ft.history.get("val_acc"))
    loss_ft = history_ft.history["loss"]
    val_loss_ft = history_ft.history["val_loss"]

    # Safely concatenate and ensure all lists are of the same length (using min_len)
    min_len = min(
        len(acc_fe) + len(acc_ft),
        len(val_acc_fe) + len(val_acc_ft),
        len(loss_fe) + len(loss_ft),
        len(val_loss_fe) + len(val_loss_ft)
    )

    acc = (acc_fe + acc_ft)[:min_len]
    val_acc = (val_acc_fe + val_acc_ft)[:min_len]
    loss = (loss_fe + loss_ft)[:min_len]
    val_loss = (val_loss_fe + val_loss_ft)[:min_len]
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    # Mark the transition point
    plt.axvline(x=initial_epochs + 1, color='r', linestyle='--', label='Fine-Tuning Start')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # Mark the transition point
    plt.axvline(x=initial_epochs + 1, color='r', linestyle='--', label='Fine-Tuning Start')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(test_gen):
    """Loads the best fine-tuned model and evaluates it on the test set."""
    
    model_filepath = "vgg16_finetuned.h5"
    if not os.path.exists(model_filepath):
        print(f"\nError: Best fine-tuned model file not found at {model_filepath}. Training likely failed.")
        return

    try:
        # Steps for test evaluation
        steps_per_epoch_test = TOTAL_TEST_IMAGES // BATCH_SIZE
        
        # Load the best saved model (saved in .h5 format)
        test_model = load_model(model_filepath)
        print("\n--- Evaluating Best Fine-Tuned Model on Test Set ---")
        
        test_loss, test_acc = test_model.evaluate(test_gen, steps=steps_per_epoch_test)
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")
        
    except Exception as e:
        print(f"\nError during final evaluation: {e}")
        print("Could not load the best model. Check your Keras/TensorFlow version compatibility.")


## Main Execution Block
# -------------------------------------------------------------
if __name__ == "__main__":
    FINAL_MODEL_FILENAME = "vgg16_finetuned.h5"
    INTERMEDIATE_MODEL_FILENAME = "vgg16_feature_extractor.h5"

    try:
        # 1. Setup paths and arguments
        ORIGINAL_DIR, NEW_BASE_DIR = setup_paths()
        
        # --- CHECK FOR EXISTING MODEL ---
        if os.path.exists(FINAL_MODEL_FILENAME):
            print(f"\n--- Detected existing model: {FINAL_MODEL_FILENAME} ---")
            print("Skipping training stages and proceeding directly to evaluation.")
            
            # Prepare data subsets (needed to create the directory structure for generators)
            prepare_data(ORIGINAL_DIR, NEW_BASE_DIR)
            
            # Create data generators
            _, _, test_generator = get_generators(NEW_BASE_DIR)

            # Final evaluation
            evaluate_model(test_generator)
            
            sys.exit(0) # Exit successfully after evaluation
            
        # --- CONTINUE TO TRAINING (If model does NOT exist) ---
        
        # 2. Prepare data subsets
        prepare_data(ORIGINAL_DIR, NEW_BASE_DIR)
        
        # 3. Build the initial model and generators
        model, conv_base = build_feature_extractor_model()
        train_generator, validation_generator, test_generator = get_generators(NEW_BASE_DIR)
        
        # --- STAGE 1: FEATURE EXTRACTION (Train only the classifier head) ---
        history_fe = run_training(
            model, 
            train_generator, 
            validation_generator,
            stage_name="Feature Extraction",
            epochs=INITIAL_EPOCHS,
            filename=INTERMEDIATE_MODEL_FILENAME # Intermediate model file
        )
        
        # 4. Load the best weights from Stage 1 before fine-tuning
        # Ensures that the best weights from the first stage are used as the starting point.
        model.load_weights(INTERMEDIATE_MODEL_FILENAME) 

        # 5. Prepare model for Fine-Tuning
        model = fine_tune_model(model, conv_base)
        
        # --- STAGE 2: FINE-TUNING (Train top VGG16 layers + classifier) ---
        history_ft = run_training(
            model, 
            train_generator, 
            validation_generator,
            stage_name="Fine-Tuning",
            epochs=TOTAL_EPOCHS, 
            # Start the fine-tuning from the next epoch index
            initial_epoch=history_fe.epoch[-1] + 1 if history_fe.epoch else 0,
            filename=FINAL_MODEL_FILENAME # Final model file
        )
        
        # 6. Plot results
        plot_history(history_ft, history_fe, TOTAL_EPOCHS, INITIAL_EPOCHS)
        
        # 7. Final evaluation
        evaluate_model(test_generator)

    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")
        sys.exit(1)
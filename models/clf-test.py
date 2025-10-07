import os
import tarfile
import urllib.request
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

load = False

# 1. Load and preprocess CIFAR-10 dataset
print("Loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Compute the global mean and standard deviation over the training set.
# Note: The computed mean and std have shape (1,1,1,3) for broadcasting.
mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
std = np.std(x_train, axis=(0, 1, 2), keepdims=True)

# Normalize (scale) the training and test images.
x_train = (x_train - mean) / std
x_test  = (x_test - mean) / std

# Split the original CIFAR-10 training data into a new training set and a validation set (90/10 split)
x_train_new, x_val, y_train_new, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42, shuffle=True
)
print("Training set shape:", x_train_new.shape)
print("Validation set shape:", x_val.shape)

# 2. Build a simple CNN model
print("\nBuilding model...")
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
#                            input_shape=(32, 32, 3)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# Load pre-trained ResNet-50 model
base_model = tf.keras.applications.VGG19(
    weights="imagenet",  # Load weights pre-trained on ImageNet
    include_top=False,   # Exclude the fully connected layers
    input_shape=(32, 32, 3)  # CIFAR-10 input size
)

# Freeze base model layers
# base_model.trainable = False  # Optional: Set to True for fine-tuning

# Add custom classification head
inputs = tf.keras.Input(shape=(32, 32, 3))
x = base_model(inputs, training=True)  # Prevent batch norm issues
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation="softmax")(x)

# Create the model
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 3. Set up EarlyStopping to halt training if validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True
)

save_path = "cifar10_classifier.h5"

if not load:
    # 4. Train the model using the training set and validate on the validation set
    print("\nTraining model...")
    history = model.fit(
        x_train_new, y_train_new,
        epochs=50,              # High maximum epoch count; training will stop early if no improvement
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping]
    )
    model.save(save_path)
else:
    model = tf.keras.models.load_model(save_path)


print(f"Classifier saved to {save_path}")

# Evaluate on the training set
print("\nEvaluating on CIFAR-10 training set...")
train_loss, train_acc = model.evaluate(x_train_new, y_train_new, verbose=0)
print("CIFAR-10 Train Accuracy: {:.4f}".format(train_acc))

# Evaluate on the validation set
print("\nEvaluating on CIFAR-10 validation set...")
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print("CIFAR-10 Validation Accuracy: {:.4f}".format(val_acc))

# Evaluate on the test set (which has not been used for training/validation)
print("\nEvaluating on CIFAR-10 test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("CIFAR-10 Test Accuracy: {:.4f}".format(test_acc))


# 5. Download and extract CIFAR-10-C (the corrupted version)
def download_and_extract_cifar10c(download_url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    tar_path = os.path.join(dest_dir, "CIFAR-10-C.tar")
    if not os.path.exists(tar_path):
        print("\nDownloading CIFAR-10-C...")
        urllib.request.urlretrieve(download_url, tar_path)
        print("Download complete.")
    else:
        print("\nCIFAR-10-C tar file already exists.")

    # Extract the tar file if the folder does not exist yet
    extract_path = os.path.join(dest_dir, "CIFAR-10-C")
    if not os.path.exists(extract_path):
        print("Extracting CIFAR-10-C...")
        with tarfile.open(tar_path, "r:") as tar:
            tar.extractall(path=dest_dir)
        print("Extraction complete.")
    else:
        print("CIFAR-10-C already extracted.")
    return extract_path

# URL for CIFAR-10-C (hosted on Zenodo)
cifar10c_url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
data_dir = "./data"  # Directory for storing downloaded files

cifar10c_folder = download_and_extract_cifar10c(cifar10c_url, data_dir)
# 6. Evaluate the model on CIFAR-10-C corruptions
print("\nEvaluating on CIFAR-10-C corruptions:")

# List all .npy files in the folder (excluding the labels file)
corruption_files = [f for f in os.listdir(cifar10c_folder) 
                    if f.endswith('.npy') and f != "labels.npy"]

# Load the CIFAR-10-C labels (ground-truth labels)
labels_path = os.path.join(cifar10c_folder, "labels.npy")
cifar10c_labels = np.load(labels_path)

# If the labels have 50,000 entries (i.e. 5 severity levels concatenated),
# reshape and select the same severity level as for the images.
if cifar10c_labels.shape[0] == 50000:
    cifar10c_labels = cifar10c_labels.reshape(5, 10000)[4]

results = {}
for file in sorted(corruption_files):
    corruption_name = file.split('.')[0]
    file_path = os.path.join(cifar10c_folder, file)
    data = np.load(file_path)
    
    # If the data has 50,000 images, it includes 5 severity levels.
    # Reshape to (5, 10000, 32, 32, 3) and select severity level 3 (index 2).
    if data.shape[0] == 50000:
        data = data.reshape(5, 10000, 32, 32, 3)
        data = data[4]
    elif data.shape[0] == 10000:
        # Data is already for one severity level.
        pass
    else:
        print(f"Unexpected shape for {file}: {data.shape}")
    
    # Normalize the images
    data = data.astype("float32") / 255.0
    # Compute the global mean and standard deviation over the training set.

    # Normalize (scale) the training and test images.
    data = (data - mean) / std

    loss, acc = model.evaluate(data, cifar10c_labels, verbose=0)
    results[corruption_name] = acc
    print("Corruption: {:20s} - Accuracy: {:.4f}".format(corruption_name, acc))

print("\nSummary of CIFAR-10-C results (severity level 3):")
for corruption, acc in results.items():
    print("  {} : {:.4f}".format(corruption, acc))

import matplotlib.pyplot as plt

# Create a dictionary that combines the accuracies for train, validation, test and all CIFAR-10-C corruption types.
accuracy_scores = {
    "train": train_acc,
    "val": val_acc,
    "test": test_acc
}
accuracy_scores.update(results)  # 'results' holds corruption accuracies from CIFAR-10-C

# Extract group names and corresponding accuracy scores.
groups = list(accuracy_scores.keys())
scores = [accuracy_scores[group] for group in groups]

# Create a bar plot.
plt.figure(figsize=(12, 6))
bars = plt.bar(groups, scores, color='skyblue', edgecolor='black')

# Annotate each bar with its accuracy value.
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.4f}",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # Offset the text by 3 points above the bar.
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.ylabel("Accuracy")
plt.title("Accuracy Scores for Train, Val, Test, and CIFAR-10-C Corruptions")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("accuracy_barplot.png")
plt.show()


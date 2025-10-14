import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

load = False

# Define the classifier-on-masked-images model as a subclass of tf.keras.Model
class clf(tf.keras.Model):
    def __init__(self, input_size, mask_interpreter, base_model=None, **kwargs):
        """
        Args:
            input_size (tuple): Shape of the input image, e.g. (32, 32, 3)
            mask_interpreter (MaskInterpreter): An instance of your MaskInterpreter model,
                which produces an importance map given an input.
        """
        super(clf, self).__init__(**kwargs)
        self.mask_interpreter = mask_interpreter
        self.mask_interpreter.trainable = False
        self.clf = self.build_simple_classifier(input_size, base_model=base_model)
    
    def compute_masked_images(self, x):
        """
        Given a batch of images x (shape: (N, H, W, C)),
        compute the importance maps using mask_interpreter (which should return a tensor of shape
        (N, H, W, 1)) and return the masked images where:
        
            x_masked = x * (1 - importance_map)
            
        Parameters:
            x: A NumPy array or tensor of shape (N, H, W, C).
            mask_interpreter: The pre-trained MaskInterpreter. Its generator is used to compute
                              the importance maps.
                              
        Returns:
            x_masked: The images masked by the inverted importance map.
        """
        # return x
        importance_mask = tf.cast(self.mask_interpreter(x),dtype=tf.float32)  # shape (B, H, W, 1)
        # importance_mask = tf.random.uniform(tf.shape(x), dtype=tf.float32)
        # Create random noise.
        normal_noise = tf.random.normal(tf.shape(importance_mask), stddev=self.mask_interpreter.noise_scale*np.random.random(), dtype=tf.float32)
        # Compute adapted (noisy) image: use importance_mask to mix x and noise.
        adapted_image = (1-importance_mask) * x + (normal_noise * importance_mask)
        return adapted_image

    def build_simple_classifier(self, input_shape, base_model=None):
        """
        Build a simple CNN classifier to be trained on masked images.
        """
        # model = keras.Sequential([
        #     keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        #     keras.layers.MaxPooling2D((2,2)),
        #     keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        #     keras.layers.MaxPooling2D((2,2)),
        #     keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(64, activation='relu'),
        #     keras.layers.Dense(10, activation='softmax')
        # ])
        # return model
        if base_model is None:
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
        return model

    def call(self, x):
        """
        Forward pass:
            - Compute the importance map using the MaskInterpreter.
            - Compute the "not important" masked image.
            - Return the classifier's predictions on the masked image.
        """
        # We assume x is a tensor or NumPy array of shape (N, H, W, C)
        x_masked = self.compute_masked_images(x)
        return self.clf(x_masked)

# -----------------------------
# Sample usage
# -----------------------------
if __name__ == "__main__":
    # For demonstration purposes, assume you have already loaded your pretrained MaskInterpreter.
    # For example, your MaskInterpreter might be defined in models/mi_clf.py and loaded as follows:
    #   from models.mi_clf import MaskInterpreter
    # And you have loaded its weights.
    #
    # Here, we assume that 'mask_interpreter' is already created and loaded.
    
    # Example: Load your MaskInterpreter (update with your actual code):
    # (This is just a placeholder; replace it with your own instantiation.)
    from models.mi_clf import MaskInterpreter
    from models.UNETO import get_unet
    
    # Load the pretrained classifier for CIFAR-10.
    classifier = keras.models.load_model('cifar10_classifier.h5')
    classifier.trainable = False

    # Create the adaptor network using your UNETO model.
    adaptor = get_unet((32,32,32), activation="sigmoid")
    adaptor.summary()
    
    # Create an instance of MaskInterpreter.
    mask_interpreter = MaskInterpreter(
        patch_size=(32, 32, 3),
        adaptor=adaptor,
        classifier=classifier,
        weighted_pcc=False,
        pcc_target=0.9
    )
    mask_interpreter.noise_scale = 0.5  # set desired noise scale
    # Build and load weights.
    mask_interpreter(np.random.random((1,32,32,3)))  # dummy forward pass to build the model
    mask_interpreter.load_weights("cifar10_mi.h5")
    
    # Now, create an instance of our classifier on masked images.
    model_clf = clf(input_size=(32,32,3), mask_interpreter=mask_interpreter, base_model=classifier)
    model_clf.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    # model_clf.summary()
    
    # Load CIFAR-10 data for training.
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    
    # Normalize images (using global mean and std computed from training set)
    mean = np.mean(x_train, axis=(0,1,2), keepdims=True)
    std  = np.std(x_train, axis=(0,1,2), keepdims=True)
    x_train = (x_train - mean) / std
    x_test  = (x_test - mean) / std
    
    # Split training data into training and validation sets.
    x_train_new, x_val, y_train_new, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42, shuffle=True)
    
    # Compute masked images on the fly using the compute_masked_images function.
    # (This will be done internally in the call() method of model_clf.)
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True
    )

    save_path = "cifar10_classifier2"
    # Train the classifier on the masked images.
    
    if not load:
        history = model_clf.fit(
            x_train_new, y_train_new,
            epochs=50,
            batch_size=128,
            validation_data=(x_val, y_val),
            callbacks = [early_stopping]
        )
    
        model_clf.save_weights(save_path,  save_format="tf")
    else:
        model_clf.load_weights(save_path)
    # Evaluate on test set.
 
 
    results = model_clf.evaluate(x_test, y_test, verbose=0)
    test_loss, test_acc = results[0], results[1]
    print("Test accuracy on masked images: {:.4f}".format(test_acc))
    

import os
import tarfile
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats

# -----------------------------
# Download and extract CIFAR-10-C if not present.
# -----------------------------
def download_and_extract_cifar10c(download_url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    tar_path = os.path.join(dest_dir, "CIFAR-10-C.tar")
    if not os.path.exists(tar_path):
        print("\nDownloading CIFAR-10-C...")
        urllib.request.urlretrieve(download_url, tar_path)
        print("Download complete.")
    else:
        print("\nCIFAR-10-C tar file already exists.")
    extract_path = os.path.join(dest_dir, "CIFAR-10-C")
    if not os.path.exists(extract_path):
        print("Extracting CIFAR-10-C...")
        with tarfile.open(tar_path, "r:") as tar:
            tar.extractall(path=dest_dir)
        print("Extraction complete.")
    else:
        print("CIFAR-10-C already extracted.")
    return extract_path

cifar10c_url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
data_dir = "./data"
cifar10c_folder = download_and_extract_cifar10c(cifar10c_url, data_dir)

# -----------------------------
# Define the class 'clf' which trains a classifier on the masked (not important) parts.
# -----------------------------
class clf_reg(tf.keras.Model):
    def __init__(self, input_size, mask_interpreter, **kwargs):
        """
        Args:
            input_size (tuple): e.g., (32, 32, 3)
            mask_interpreter: an instance of your MaskInterpreter (pretrained) that outputs an importance map.
        """
        super(clf, self).__init__(**kwargs)
        self.mask_interpreter = mask_interpreter
        self.clf = self.build_simple_classifier(input_size)
    
    def compute_masked_images(self, x, mask_interpreter):
        """
        Given a batch of images x (shape: (N, H, W, C)), compute importance maps on the fly
        using mask_interpreter, then return masked images:
        
           x_masked = x * (1 - importance_map)
        """
        # Compute importance maps via the MaskInterpreter’s generator.
        importance_map = mask_interpreter(x)  # shape: (N, H, W, 1)
        if isinstance(importance_map, tf.Tensor):
            importance_map = importance_map.numpy()
        # If x has 3 channels and importance_map is single-channel, tile it.
        if importance_map.shape[-1] == 1 and x.shape[-1] != 1:
            importance_map = np.tile(importance_map, (1, 1, 1, x.shape[-1]))
        x_masked = x * (1 - importance_map)
        return x_masked
    
    def build_simple_classifier(self, input_shape):
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        return model
    
    def call(self, x):
        """
        Forward pass: Compute masked images and then output predictions.
        """
        x_masked = self.compute_masked_images(x, self.mask_interpreter)
        return self.clf(x_masked)

# -----------------------------
# Majority Vote Function: Combine predictions from two models.
# -----------------------------
def majority_vote(preds1, preds2):
    """
    Given two arrays of predictions (shape: (N, num_classes)) from two models,
    return an array of final predicted labels using majority vote:
      - If both agree, use that class.
      - Otherwise, use the class from the average of the probability vectors.
    """
    class1 = np.argmax(preds1, axis=1)
    class2 = np.argmax(preds2, axis=1)
    avg_preds = (preds1 + preds2) / 2.0
    avg_class = np.argmax(avg_preds, axis=1)
    final_pred = np.where(class1 == class2, class1, avg_class)
    return final_pred

# -----------------------------
# Evaluation Function: Compute accuracies for three methods.
# -----------------------------
def evaluate_methods(x, y, classifier, model_clf):
    """
    For dataset (x, y), compute:
       - Accuracy of the original classifier.
       - Accuracy of model_clf (trained on masked images).
       - Accuracy of majority vote between the two.
    Returns a tuple of (acc_classifier, acc_model_clf, acc_majority)
    """
    preds_class = classifier.predict(x)      # shape (N, num_classes)
    preds_masked = model_clf.predict(x)        # internally computes masked images
    majority_preds = majority_vote(preds_class, preds_masked)
    true_labels = y.squeeze()
    
    acc_class = np.mean(np.argmax(preds_class, axis=1) == true_labels)
    acc_masked = np.mean(np.argmax(preds_masked, axis=1) == true_labels)
    acc_majority = np.mean(majority_preds == true_labels)
    
    return acc_class, acc_masked, acc_majority
    
# (Assume model_clf is already trained. Otherwise, train it:)
# history = model_clf.fit(x_train_new, y_train_new, epochs=20, batch_size=64, validation_data=(x_val, y_val))

# --- 4. Evaluate on CIFAR-10 Train, Val, and Test sets ---
print("Evaluating on CIFAR-10 datasets...")
acc_train = evaluate_methods(x_train_new, y_train_new, classifier, model_clf)
acc_val   = evaluate_methods(x_val, y_val, classifier, model_clf)
acc_test  = evaluate_methods(x_test, y_test, classifier, model_clf)
print("CIFAR-10 Accuracy:")
print("  Original classifier: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(acc_train[0], acc_val[0], acc_test[0]))
print("  Masked classifier:   Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(acc_train[1], acc_val[1], acc_test[1]))
print("  Majority vote:       Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(acc_train[2], acc_val[2], acc_test[2]))

# --- 5. Process CIFAR-10-C corruptions ---
print("Processing CIFAR-10-C corruptions...")
cifar10c_labels_path = os.path.join(cifar10c_folder, "labels.npy")
cifar10c_labels = np.load(cifar10c_labels_path)
# If there are 50,000 labels (5 severity levels concatenated), choose one severity level.
if cifar10c_labels.shape[0] == 50000:
    # Here we choose severity level 3 (index 2), adjust as needed.
    cifar10c_labels = cifar10c_labels.reshape(5, 10000)[4]

corruption_files = sorted([f for f in os.listdir(cifar10c_folder) if f.endswith('.npy') and f != "labels.npy"])
corruption_acc = {}  # To store accuracies for each corruption group for each method.
for file in corruption_files:
    corruption_name = file.split('.')[0]
    file_path = os.path.join(cifar10c_folder, file)
    data = np.load(file_path)
    if data.shape[0] == 50000:
        # Reshape to (5, 10000, 32, 32, 3) and select one severity level, e.g., index 2.
        data = data.reshape(5, 10000, 32, 32, 3)[4]
    elif data.shape[0] == 10000:
        pass
    else:
        print(f"Unexpected shape for {file}: {data.shape}")
    data = data.astype("float32") / 255.0
    data_norm = (data - mean) / std
    
    # Use only a subset for speed.
    num_subset = min(1000, data_norm.shape[0])
    x_corr = data_norm[:num_subset]
    y_corr = cifar10c_labels[:num_subset]
    
    acc_corr = evaluate_methods(x_corr, y_corr, classifier, model_clf)
    corruption_acc[corruption_name] = acc_corr
    print("Corruption: {:20s} - Classifier: {:.4f}, Masked: {:.4f}, Majority: {:.4f}".format(
        corruption_name, acc_corr[0], acc_corr[1], acc_corr[2]
    ))

# --- 6. Plot Bar Plot of All Datasets Accuracy ---
# We'll build a grouped bar plot comparing the three methods (Original, Masked, Majority) on each group.
# Groups: "Train", "Val", "Test", plus each corruption.
groups = ["Train", "Val", "Test"] + sorted(corruption_acc.keys())
# For each group, store a tuple of accuracies: (original, masked, majority)
accuracy_scores = {}
accuracy_scores["Train"] = acc_train
accuracy_scores["Val"]   = acc_val
accuracy_scores["Test"]  = acc_test
for key in corruption_acc:
    accuracy_scores[key] = corruption_acc[key]

# For plotting, we create three lists (one per method) with accuracies in the same group order.
orig_acc_list = []
masked_acc_list = []
majority_acc_list = []
for group in groups:
    orig, masked, majority = accuracy_scores[group]
    orig_acc_list.append(orig)
    masked_acc_list.append(masked)
    majority_acc_list.append(majority)

# Create grouped bar plot.
x = np.arange(len(groups))
bar_width = 0.25

plt.figure(figsize=(14, 7))
bars1 = plt.bar(x - bar_width, orig_acc_list, bar_width, label="Original Classifier")
bars2 = plt.bar(x, masked_acc_list, bar_width, label="Masked Classifier")
bars3 = plt.bar(x + bar_width, majority_acc_list, bar_width, label="Majority Vote")

plt.ylabel("Accuracy")
plt.title("Accuracy Comparison on CIFAR-10 and CIFAR-10-C Datasets")
plt.xticks(x, groups, rotation=45, ha='right')
plt.ylim([0,1])
plt.legend()

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

plt.tight_layout()
plt.savefig("combined_accuracy_barplot.png")
plt.show()


# Compute mean improvements: (Masked - Original) and (Majority - Original)
orig_acc_list = []
masked_acc_list = []
majority_acc_list = []
masked_improvement = []
majority_improvement = []

for key, (orig, masked, majority) in accuracy_scores.items():
    orig_acc_list.append(orig)
    masked_acc_list.append(masked)
    majority_acc_list.append(majority)
    masked_improvement.append(masked - orig)
    majority_improvement.append(majority - orig)
    
mean_masked_improvement = np.mean(masked_improvement)
mean_majority_improvement = np.mean(majority_improvement)

# Plot mean improvements
labels = ["Masked - Original", "Majority - Original"]
values = [mean_masked_improvement, mean_majority_improvement]

plt.figure(figsize=(6, 5))
bars = plt.bar(labels, values, color=['blue', 'green'])

plt.ylabel("Mean Accuracy Improvement")
plt.title("Mean Accuracy Improvement Over All Datasets")
plt.axhline(y=0, color='gray', linestyle='--')

# Annotate bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.3f}",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.tight_layout()
plt.savefig("improvement_barplot.png")
plt.show()




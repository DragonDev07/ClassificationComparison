from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import numpy as np


# `load_mnist(test_size=0.2, val_size=0.2, random_state=42) -->
#   - Loads the mnist dataset and returns it as a dictionary with keys 'train', 'val', and 'test' splits.
def load_mnist(test_size=0.2, val_size=0.2, random_state=42):
    mnist = fetch_openml("mnist_784", version=1)
    X, y = mnist["data"], mnist["target"].astype(int)

    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


# `load_tiny_imagenet(test_size=0.2, val_size=0.2, random_state=42) -->
#   - Loads the tiny-imagenet dataset and returns it as a dictionary with keys 'train', 'val', and 'test' splits.
def load_tiny_imagenet(test_size=0.2, val_size=0.2, random_state=42):
    # Load the dataset from Hugging Face
    dataset = load_dataset("zh-plus/tiny-imagenet")

    # Convert images to consistent format (RGB)
    def process_image(img):
        # Convert to RGB if not already
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Resize if needed (shouldn't be necessary for Tiny ImageNet)
        return np.array(img)

    # Extract images and labels, ensuring consistent format
    X_full = np.array([process_image(img["image"]) for img in dataset["train"]])
    y_full = np.array([img["label"] for img in dataset["train"]])

    # Reshape images to be flat
    # Original shape: (N, H, W, C) -> New shape: (N, H*W*C)
    X_full_flat = X_full.reshape(X_full.shape[0], -1)

    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full_flat, y_full, test_size=test_size, random_state=random_state
    )

    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    # Store original image shape for reshaping later if needed
    original_shape = X_full.shape[1:]

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "metadata": {
            "original_shape": original_shape,
            "num_classes": len(np.unique(y_full)),
            "dataset_name": "tiny-imagenet",
        },
    }

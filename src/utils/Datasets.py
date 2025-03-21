from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
import numpy as np


# `load_mnist(test_size=0.2, val_size=0.2, random_state=42) -->
#   - Loads the mnist dataset and returns it as a dictionary with keys 'train', 'val', and 'test' splits.
def load_mnist(test_size=0.2, val_size=0.2, random_state=42):
    mnist = fetch_openml("mnist_784", version=1)
    X = (
        mnist["data"].to_numpy()
        if hasattr(mnist["data"], "to_numpy")
        else mnist["data"]
    )
    y = (
        mnist["target"].astype(int).to_numpy()
        if hasattr(mnist["target"], "to_numpy")
        else mnist["target"].astype(int)
    )

    # Analyze dataset properties
    num_samples = X.shape[0]
    pixels = X.shape[1]
    side_length = int(np.sqrt(pixels))
    input_shape = (1, side_length, side_length)  # (channels, height, width)
    num_classes = len(np.unique(y))

    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    return {
        "name": "MNIST",
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "metadata": {
            "input_shape": input_shape,
            "num_classes": num_classes,
            "num_samples": num_samples,
            "dataset_name": "MNIST",
        },
    }


# `load_tiny_imagenet(test_size=0.2, val_size=0.2, random_state=42) -->
#   - Loads the tiny-imagenet dataset and returns it as a dictionary with keys 'train', 'val', and 'test' splits.
def load_tiny_imagenet(test_size=0.2, val_size=0.2, random_state=42):
    dataset = load_dataset("zh-plus/tiny-imagenet")

    def process_image(img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    X_full = np.array([process_image(img["image"]) for img in dataset["train"]])
    y_full = np.array([img["label"] for img in dataset["train"]])

    # Analyze dataset properties
    num_samples = X_full.shape[0]
    input_shape = (3, X_full.shape[1], X_full.shape[2])  # (channels, height, width)
    num_classes = len(np.unique(y_full))

    # Reshape images to be flat
    X_full_flat = X_full.reshape(X_full.shape[0], -1)

    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full_flat, y_full, test_size=test_size, random_state=random_state
    )

    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    return {
        "name": "tiny-imagenet",
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "metadata": {
            "input_shape": input_shape,
            "num_classes": num_classes,
            "num_samples": num_samples,
            "dataset_name": "tiny-imagenet",
        },
    }

import time
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import umap


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Layer 1: First Convolutional Layer
        # Input channels: 1 (grayscale)
        # Output channels: 32
        # Kernel size: 3x3
        # Stride: 1
        # No padding
        # Output size: (28-3+1)x(28-3+1) = 26x26
        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        # Layer 2: Second Convolutional Layer
        # Input channels: 32 (from previous layer)
        # Output channels: 64
        # Kernel size: 3x3
        # Stride: 1
        # No padding
        # Output size: (26-3+1)x(26-3+1) = 24x24
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Dropout Layers
        # Layer 3: First Dropout Layer (25% dropout)
        self.dropout1 = nn.Dropout(0.25)
        # Layer 5: Second Dropout Layer (50% dropout)
        self.dropout2 = nn.Dropout(0.5)

        # Layer 4: First Fully Connected Layer
        # Input size: 9216 (64 channels * 12 * 12)
        # Output size: 128 neurons
        self.fc1 = nn.Linear(9216, 128)

        # Layer 6: Second Fully Connected Layer (Output Layer)
        # Input size: 128 neurons
        # Output size: 10 (number of classes in MNIST)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First Convolutional Block
        x = self.conv1(x)  # Apply first conv layer
        x = F.relu(x)  # Apply ReLU activation

        # Second Convolutional Block
        x = self.conv2(x)  # Apply second conv layer
        x = F.relu(x)  # Apply ReLU activation
        x = F.max_pool2d(x, 2)  # Apply max pooling with 2x2 kernel
        # Output size: 12x12

        # First Dropout
        x = self.dropout1(x)  # Apply 25% dropout

        # Flatten Layer
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # Size: batch_size x 9216

        # First Fully Connected Layer
        x = self.fc1(x)  # Apply first FC layer
        x = F.relu(x)  # Apply ReLU activation

        # Second Dropout
        x = self.dropout2(x)  # Apply 50% dropout

        # Output Layer
        x = self.fc2(x)  # Apply second FC layer

        # Apply log softmax for probability distribution
        # dim=1 applies softmax across the classes
        output = F.log_softmax(x, dim=1)

        return output


# `load_dataset` -->
#   - Loads the MNIST dataset from OpenML
#   - Splits the dataset into training, validation, and testing sets
def load_dataset(test_size=0.2, val_size=0.2, random_state=42):
    print("Loading MNIST dataset...")

    mnist = fetch_openml("mnist_784", version=1)
    X = mnist["data"].to_numpy().astype("float32")  # Convert DataFrame to numpy array
    y = mnist["target"].to_numpy().astype(int)  # Convert Series to numpy array

    # Normalize the data
    X = X / 255.0

    # Reshape data for CNN
    X = X.reshape(-1, 1, 28, 28)

    # First split: training + validation vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: training vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# `train_cnn` -->
#   - Trains a CNN model using PyTorch
#   - Returns the trained CNN model
def train_cnn(X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
    print("Training CNN model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train.astype(int))  # Ensure integer type
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val.astype(int))  # Ensure integer type

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            pbar.set_postfix({"loss": f"{train_loss / (batch_idx + 1):.4f}"})

    return model


# `evaluate_cnn_accuracy` -->
#   - Evaluates the CNN model's accuracy on test data
#   - Returns the accuracy score
def evaluate_cnn_accuracy(model, X_test, y_test, batch_size=64):
    print("Evaluating CNN model accuracy...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


# `measure_prediction_speed` -->
#   - Measures the average prediction speed of the CNN model
#   - Returns the average prediction speed
def measure_prediction_speed(model, X_test, n_trials=100):
    print("Measuring prediction speed...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_test = torch.FloatTensor(X_test)
    total_time = 0

    with torch.no_grad():
        for _ in tqdm(range(n_trials)):
            start_time = time.time()
            model(X_test.to(device))
            end_time = time.time()
            total_time += end_time - start_time

    return total_time / n_trials


# `visualize_umap` -->
#   - Generates a UMAP visualization of the predictions
#   - Saves the visualization as a PNG file
def visualize_umap(X, predictions, title="UMAP Visualization of Predictions"):
    print("Generating & Saving UMAP visualization...")

    # Flatten the input data
    X_flat = X.reshape(X.shape[0], -1)

    # Perform UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        n_jobs=-1,
    )
    X_umap = reducer.fit_transform(X_flat)

    plt.figure(figsize=(12, 10), dpi=300)
    scatter = sns.scatterplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=predictions,
        palette="tab10",
        alpha=0.8,
        s=100,
        edgecolor="black",
        linewidth=0.5,
        legend="full",
    )

    scatter.legend(title="Class", fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/cnn_umap.png", dpi=300, bbox_inches="tight")
    plt.close()


# `save_model` -->
#   - Saves a model to a file using joblib
def save_model(model, filename="cnn_model.joblib"):
    print(f"Saving model to {filename}...")
    joblib.dump(model, filename)


# `load_model` -->
#   - Loads a model from a file using joblib
def load_model(filename="trained_models/cnn_model.joblib"):
    print(f"Loading model from {filename}...")
    return joblib.load(filename)


# Main function
def main():
    model_filename = "trained_models/cnn_model.joblib"

    # Check if we have a saved model
    if os.path.exists(model_filename):
        # Load existing model
        model = load_model(model_filename)

        # Load only test data for evaluation
        _, _, X_test, _, _, y_test = load_dataset()
    else:
        # Load full dataset and train new model
        X_train, X_val, X_test, y_train, y_val, y_test = load_dataset()

        # Train the CNN model
        model = train_cnn(X_train, y_train, X_val, y_val, epochs=20, batch_size=64)

        # Save the model
        save_model(model, model_filename)

    # Get predictions for visualization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_tensor).argmax(dim=1).cpu().numpy()

    # Generate UMAP visualization
    visualize_umap(X_test, predictions, "UMAP Visualization of MNIST Predictions (CNN)")

    # Evaluate the CNN model
    accuracy = evaluate_cnn_accuracy(model, X_test, y_test)
    prediction_speed = measure_prediction_speed(model, X_test)

    # Print Results of the classifier
    print("----------- CNN Model Results -----------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Prediction Speed: {prediction_speed:.4f} seconds")


if __name__ == "__main__":
    main()

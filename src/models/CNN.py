import time
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import umap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class CNNModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNNModel, self).__init__()

        # Layer 1: First Convolutional Layer
        # Output channels: 32
        # Kernel size: 3x3
        # Stride: 1
        # Padding: 1
        # Output size: (28-3+1)x(28-3+1) = 26x26
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, padding=1)

        # Layer 2: Second Convolutional Layer
        # Input channels: 32 (from previous layer)
        # Output channels: 64
        # Kernel size: 3x3
        # Stride: 1
        # Padding: 1
        # Output size: (26-3+1)x(26-3+1) = 24x24
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)

        # Dropout Layers
        # Layer 3: First Dropout Layer (25% dropout)
        self.dropout1 = nn.Dropout(0.25)
        # Layer 5: Second Dropout Layer (50% dropout)
        self.dropout2 = nn.Dropout(0.5)

        # Layer 4: First Fully Connected Layer
        # Size will be calculated in forward pass
        self.fc1 = None

        # Layer 6: Second Fully Connected Layer (Output Layer)
        # Input size: 128 neurons
        # Output size: Number of Classes
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # First Convolutional Block
        x = self.conv1(x)  # Apply first conv layer
        x = F.relu(x)  # Apply ReLU activation

        # Second Convolutional Block
        x = self.conv2(x)  # Apply second conv layer
        x = F.relu(x)  # Apply ReLU activation
        x = F.max_pool2d(x, 2)  # Apply max pooling with 2x2 kernel

        # First Dropout
        x = self.dropout1(x)  # Apply 25% dropout

        # Flatten Layer
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Initialize fc1 if not done yet (to handle different input sizes)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)

        # First Fully Connected Layer
        x = self.fc1(x)  # Apply first FC layer
        x = F.relu(x)  # Apply ReLU activation

        # Second Dropout
        x = self.dropout2(x)  # Apply 50% dropout

        # Output Layer
        x = self.fc2(x)  # Apply second FC layer
        output = F.log_softmax(x, dim=1)

        return output


class CNN:
    def __init__(self, epochs=10, batch_size=64):
        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            print(f"GPU: {gpu}")
        self.dataset_name = "Unknown"
        self.base_path = "../out"
        self.input_shape = None
        self.num_classes = None

    # `_generate_paths()` -->
    #   - Generate the paths for saving models and visualizations
    def _generate_paths(self):
        # Create the base directories if they don't exist
        model_dir = f"{self.base_path}/trained_models"
        viz_dir = f"{self.base_path}/visualizations"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        # Generate the base name for files
        base_name = f"CNN_{self.epochs}epochs_{self.dataset_name}"

        # Generate complete paths
        model_path = f"{model_dir}/{base_name}.joblib"
        umap_path = f"{viz_dir}/{base_name}_umap.png"

        return model_path, umap_path

    # `_preprocess_data(X)` -->
    #   - Preprocess data for training
    def _preprocess_data(self, X):
        # Convert to numpy array if not already
        X = np.array(X)

        # If data is flat (like MNIST), reshape appropriately
        if len(X.shape) == 2:  # Flat data
            channels, height, width = self.input_shape
            X = X.reshape(-1, channels, height, width)
        else:  # Image format (N, H, W, C) or (N, C, H, W)
            if X.shape[-1] == self.input_shape[0]:  # If channels are last (N, H, W, C)
                # Use np.transpose instead of .transpose
                X = np.moveaxis(X, -1, 1)  # Move channels from last axis to axis 1

        # Normalize pixel values to [0,1]
        X = X.astype(np.float32) / 255.0

        return X

    # `train(X_train, y_train, X_val, y_val, dataset_name='Unknown')` -->
    #   - Train the CNN model
    def train(self, X_train, y_train, X_val, y_val, metadata):
        # Set metadata from dataset
        self.dataset_name = metadata["dataset_name"]
        self.input_shape = metadata["input_shape"]
        self.num_classes = metadata["num_classes"]

        # Initialize model with correct parameters
        self.model = CNNModel(
            in_channels=self.input_shape[0], num_classes=self.num_classes
        ).to(self.device)

        # Preprocess data
        X_train = self._preprocess_data(X_train)

        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train.astype(int))

        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                pbar.set_postfix({"loss": f"{train_loss / (batch_idx + 1):.4f}"})

        return self

    # `evaluate(X_test, y_test)` -->
    #   - Evaluate the model accuracy and prediction speed
    def evaluate(self, X_test, y_test):
        self.model.eval()

        # Preprocess the data
        X_test = self._preprocess_data(X_test)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        # Create data loader
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # Initialize lists to store predictions and true labels
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)

                # Collect predictions and targets
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Calculate metrics
        results = {
            "accuracy": accuracy_score(all_targets, all_predictions),
            "precision": precision_score(
                all_targets, all_predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                all_targets, all_predictions, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                all_targets, all_predictions, average="weighted", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(all_targets, all_predictions),
            "classification_report": classification_report(
                all_targets, all_predictions, zero_division=0
            ),
            "prediction_speed": self._measure_prediction_speed(X_test),
        }

        return results

    # `predict(X)` -->
    #   - Make predictions using the trained model
    def predict(self, X, batch_size=None):
        self.model.eval()
        X = self._preprocess_data(X)
        X = torch.FloatTensor(X)
        dataset = TensorDataset(X)
        dataloader = DataLoader(
            dataset, batch_size=batch_size if batch_size else self.batch_size
        )
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(self.device)
                output = self.model(batch)
                pred = output.argmax(dim=1).cpu().numpy()
                predictions.append(pred)

        return np.concatenate(predictions)

    # `generate_umap(X, predictions, save_path=None)` -->
    #   - Generate UMAP visualization using seaborn
    def generate_umap(self, X, predictions):
        # Get save path and setup number of classes
        _, umap_path = self._generate_paths()
        unique_classes = np.unique(predictions)
        n_classes = len(unique_classes)

        # Initialize UMAP reducer and transform data
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        umap_result = reducer.fit_transform(X)

        # Setup style once
        sns.set_style(
            "darkgrid",
            {
                "axes.edgecolor": "0.2",
                "axes.linewidth": 1.5,
                "grid.color": "0.75",
                "grid.linestyle": "-",
            },
        )

        # Create figure and plot
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

        # Plot data
        sns.scatterplot(
            x=umap_result[:, 0],
            y=umap_result[:, 1],
            hue=predictions,
            palette="husl" if n_classes > 20 else "tab10",
            alpha=0.8,
            s=100,
            edgecolor="black",
            linewidth=0.5,
            legend="auto",
            ax=ax,
        )

        # Style adjustments
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)

        # Set title and labels with enhanced styling
        ax.set_title(
            "UMAP Visualization of Predictions", pad=20, fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("UMAP Component 1", fontsize=12)
        ax.set_ylabel("UMAP Component 2", fontsize=12)

        # Add legend with enhanced styling
        ax.legend(title="Class", fontsize=12, title_fontsize=12)

        # Save plot
        if umap_path:
            plt.savefig(umap_path, bbox_inches="tight", dpi=300)
        plt.close()

    # `save()` -->
    #   - Save the model to file
    def save(self):
        model_path, _ = self._generate_paths()
        # Save both model and configuration
        config = {
            "model": self.model,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
        }
        joblib.dump(config, model_path)
        return model_path

    # `load()` -->
    #   - Load the model from file
    def load(self, filename=None):
        if filename is None:
            model_path, _ = self._generate_paths()
            filename = model_path

        # Load configuration
        config = joblib.load(filename)

        # Set model and configuration
        self.model = config["model"]
        self.input_shape = config["input_shape"]
        self.num_classes = config["num_classes"]

        print("Loaded model configuration:")
        print(f"Input shape: {self.input_shape}")
        print(f"Number of classes: {self.num_classes}")

        return self

    # `_evaluate_accuracy(X_test, y_test)` -->
    #   - Helper method to evaluate model accuracy
    def _evaluate_accuracy(self, X_test, y_test):
        self.model.eval()

        # Preprocess the data
        X_test = self._preprocess_data(X_test)

        # Convert to PyTorch tensors
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return correct / total

    # `_measure_prediction_speed(X_test, n_trials=100)` -->
    #   - Helper method to measure prediction speed
    def _measure_prediction_speed(self, X_test, n_trials=100):
        self.model.eval()

        # Preprocess the data first
        X_test = self._preprocess_data(X_test)
        X_test = torch.FloatTensor(X_test)

        # Create DataLoader for batched processing
        dataset = TensorDataset(X_test)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        total_time = 0
        with torch.no_grad():
            for _ in tqdm(range(n_trials)):
                batch_times = []
                for batch in dataloader:
                    batch = batch[0].to(self.device)
                    start_time = time.time()
                    self.model(batch)
                    batch_times.append(time.time() - start_time)
                total_time += sum(batch_times)

        return total_time / n_trials

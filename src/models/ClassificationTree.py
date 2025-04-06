import time
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import umap
import numpy as np


class ClassificationTree:
    def __init__(self, base_max_depth=None, random_state=42):
        self.base_max_depth = base_max_depth
        self.random_state = random_state
        self.model = None
        self.dataset_name = "Unknown"
        self.is_grid_search = False
        self.base_path = "../out"

    # `_generate_paths()` -->
    #   - Generate the paths for saving models and visualizations
    def _generate_paths(self):
        # Create the base directories if they don't exist
        model_dir = f"{self.base_path}/trained_models"
        viz_dir = f"{self.base_path}/visualizations"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        # Generate the base name for files
        base_name = f"ClassificationTree{'_GridSearch' if self.is_grid_search else ''}_{self.dataset_name}"

        # Generate complete paths
        model_path = f"{model_dir}/{base_name}.joblib"
        umap_path = f"{viz_dir}/{base_name}_umap.png"

        return model_path, umap_path

    # `train(X_train, y_train) -->
    #   - Directly train the model without grid search
    def train(self, X_train, y_train):
        self.is_grid_search = False

        # Initialize the model
        self.model = DecisionTreeClassifier(
            max_depth=self.base_max_depth, random_state=self.random_state
        )

        # Fit the model
        self.model.fit(X_train, y_train)

        # Return self
        return self

    # `grid_train(X_train, y_train) -->
    #   - Train the model using grid search
    def grid_train(self, X_train, y_train, n_jobs=None):
        self.is_grid_search = True

        # Define the parameter grid
        param_grid = {
            "max_depth": [2, 4, 8, 16, 32, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion": ["gini", "entropy"],
        }

        # Initialize the base classifier
        base_classifier = DecisionTreeClassifier(
            max_depth=self.base_max_depth, random_state=self.random_state
        )

        # Initialize the grid search
        grid_search = GridSearchCV(
            estimator=base_classifier,
            param_grid=param_grid,
            cv=5,
            verbose=3,
            n_jobs=n_jobs,
        )

        # Fit the grid search
        grid_search.fit(X_train, y_train)

        # Get the best model
        self.model = grid_search.best_estimator_

        # Return self
        return self

    # `prune(X_val, y_val) -->
    #   - Prune the model using cost complexity pruning
    def prune(self, X_val, y_val):
        # Get the pruning path
        pruning_path = self.model.cost_complexity_pruning_path(X_val, y_val)

        # Get the alphas and impurities (unused)
        alphas, _ = pruning_path.ccp_alphas, pruning_path.impurities

        # Initialize variables to store the best classifier and its cross-validation score
        best_classifier = None
        best_cv_mean_score = 0

        # Loop through the alphas and prune the model
        for alpha in tqdm(alphas):
            # Initialize the pruned classifier
            pruned_classifier = DecisionTreeClassifier(
                random_state=self.random_state, ccp_alpha=alpha
            )

            # Fit the pruned classifier
            pruned_classifier.fit(X_val, y_val)

            # Evaluate the pruned classifier
            cv_score = pruned_classifier.score(X_val, y_val)

            # Check if the current classifier has a better cross-validation score
            if cv_score > best_cv_mean_score:
                # Update the best classifier and its cross-validation score
                best_classifier = pruned_classifier
                best_cv_mean_score = cv_score

        # Return self
        self.model = best_classifier
        return self

    # `evaluate(X_test, y_test, cv=5) -->
    #   - Evaluate the model using:
    #       - Cross-validation score
    #       - Accuracy score
    #       - Prediction speed
    def evaluate(self, X_test, y_test):
        # Get predictions
        predictions = self.predict(X_test)

        # Calculate metrics
        results = {
            # Basic Metrics
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted"),
            "recall": recall_score(y_test, predictions, average="weighted"),
            "f1": f1_score(y_test, predictions, average="weighted"),
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "classification_report": classification_report(y_test, predictions),
            "prediction_speed": self._measure_prediction_speed(X_test),
        }

        return results

    # `predict(X)` -->
    #   - Predict the class labels for the given data
    def predict(self, X):
        return self.model.predict(X)

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

    # `save(self, filename)` -->
    #    - Save the model to a file
    def save(self):
        # Generate model path
        model_path, _ = self._generate_paths()

        # Save the model to a file
        joblib.dump(self.model, model_path)
        return model_path

    # `load(self, filename)` -->
    #    - Load the model from a file
    def load(self, filename=None):
        if filename is None:
            # Generate paths if filename is not provided
            model_path, *_ = self._generate_paths()
            filename = model_path

        # Load the model from the file
        self.model = joblib.load(filename)
        return self

    # `_measure_prediction_speed(self, X_test, n_trials=100)` -->
    #    - Helper function to measure average prediction speed
    def _measure_prediction_speed(self, X_test, n_trials=100):
        total_time = 0
        for _ in tqdm(range(n_trials)):
            start_time = time.time()
            self.model.predict(X_test)
            total_time += time.time() - start_time
        return total_time / n_trials

import time
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import umap


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
    def evaluate(self, X_test, y_test, cv=5):
        cv_score = cross_val_score(self.model, X_test, y_test, cv=cv).mean()
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        prediction_speed = self._measure_prediction_speed(X_test)

        return {
            "cv_score": cv_score,
            "accuracy": accuracy,
            "prediction_speed": prediction_speed,
        }

    # `predict(X)` -->
    #   - Predict the class labels for the given data
    def predict(self, X):
        return self.model.predict(X)

    # `generate_umap(X, predictions, save_path=None)` -->
    #   - Generate UMAP visualization
    def generate_umap(self, X, predictions):
        # Get save path
        _, umap_path = self._generate_paths()

        # Initialize UMAP reducer
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, n_jobs=-1)
        X_umap = reducer.fit_transform(X)

        # Initialize plot
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

        # Initialize legend
        scatter.legend(title="Class", fontsize=12)
        plt.title("UMAP Visualization of Predictions", fontsize=14, pad=20)
        plt.xlabel("UMAP Component 1", fontsize=12)
        plt.ylabel("UMAP Component 2", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if umap_path:
            # Save plot
            plt.savefig(umap_path, dpi=300, bbox_inches="tight")
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

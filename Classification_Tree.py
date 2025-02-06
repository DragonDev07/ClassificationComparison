import time
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import umap


# `load_dataset` -->
#   - Loads the MNIST dataset from OpenML
#   - Splits the dataset into training, validation, and testing sets
def load_dataset(test_size=0.2, val_size=0.2, random_state=42):
    print("Loading MNIST dataset...")

    mnist = fetch_openml("mnist_784", version=1)
    X, y = mnist["data"], mnist["target"].astype(int)

    # First split: training + validation vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: training vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# `train_decision_tree` -->
#   - Trains a decision tree classifier using GridSearchCV to find the best hyperparameters
#   - Returns the best decision tree classifier
def train_decision_tree(X_train, y_train, base_max_depth=None, random_state=42):
    print("Training decision tree classifier and finding best hyperparameters...")

    # Define the hyperparameters to tune
    param_grid = {
        "max_depth": [2, 4, 8, 16, 32, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8],
        "criterion": ["gini", "entropy"],
    }

    base_classifier = DecisionTreeClassifier(
        max_depth=base_max_depth, random_state=random_state
    )

    # Create GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(
        estimator=base_classifier,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=3,
    )

    grid_search.fit(X_train, y_train)

    print(f"--> Best parameters found: {grid_search.best_params_}")
    print(f"--> Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# `prune_decision_tree` -->
#   - Prunes a decision tree classifier using cost complexity pruning
#   - Returns the best pruned decision tree classifier
def prune_decision_tree(classifier, X_val, y_val, random_state=42):
    # Get the cost complexity pruning path
    pruning_path = classifier.cost_complexity_pruning_path(X_val, y_val)

    # Get the alphas and impurities
    alphas, _impurities = pruning_path.ccp_alphas, pruning_path.impurities

    # Train a decision tree classifier for each alpha
    print(f"Creating and training pruned classifiers for {len(alphas)} alphas...")
    pruned_classifiers = []
    for alpha in tqdm(alphas):
        pruned_classifier = DecisionTreeClassifier(
            random_state=random_state, ccp_alpha=alpha
        )
        pruned_classifier.fit(X_val, y_val)
        pruned_classifiers.append(pruned_classifier)

    # Find the best pruned classifier tree
    print(
        f"Finding best pruned classifier out of {len(pruned_classifiers)} pruned classifiers..."
    )
    best_classifier = None
    best_cv_mean_score = 0
    for pruned_classifier in tqdm(pruned_classifiers):
        cv_score = pruned_classifier.score(X_val, y_val)

        if cv_score > best_cv_mean_score:
            print(
                f"--> Found better pruned classifier with cross-validation score: {cv_score}",
            )
            best_classifier = pruned_classifier
            best_cv_mean_score = cv_score

    return best_classifier


# `evaluate_decision_tree_cv` -->
#   - Evaluates a decision tree classifier using cross-validation
#   - Returns the mean cross-validation score
def evaluate_decision_tree_cv(classifier, X_test, y_test, cv=5):
    print("Evaluating decision tree classifier (cross-validation)...")
    cv_scores = cross_val_score(classifier, X_test, y_test, cv=cv)
    return cv_scores.mean()


# `evaluate_decision_tree_accuracy` -->
#   - Evaluates a decision tree classifier using accuracy score
#   - Returns the accuracy score
def evaluate_decision_tree_accuracy(classifier, X_test, y_test):
    print("Evaluating decision tree classifier (accuracy score)...")
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


# `measure_prediction_speed` -->
#   - Measures the average prediction speed of a classifier
#   - Returns the average prediction speed
def measure_prediction_speed(classifier, X_test, n_trials=100):
    print("Measuring prediction speed...")
    total_time = 0

    for _ in tqdm(range(n_trials)):
        start_time = time.time()
        classifier.predict(X_test)
        end_time = time.time()
        total_time += end_time - start_time

    return total_time / n_trials


# `visualize_umap` -->
#   - Generates a UMAP visualization of the predictions
#   - Saves the visualization as a PNG file
def visualize_umap(X, predictions, title="UMAP Visualization of Predictions"):
    print("Generating & Saving UMAP visualization...")

    # Perform UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        n_jobs=-1,  # Enable parallel processing
    )
    X_umap = reducer.fit_transform(X)

    # Create high quality scatter plot using Seaborn
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
    plt.savefig("visualizations/decision_tree_umap.png", dpi=300, bbox_inches="tight")
    plt.close()


# `save_model` -->
#   - Saves a model to a file using joblib
def save_model(model, filename="decision_tree_model.joblib"):
    print(f"Saving model to {filename}...")
    joblib.dump(model, filename)


# `load_model` -->
#   - Loads a model from a file using joblib
def load_model(filename="trained_models/decision_tree_model.joblib"):
    print(f"Loading model from {filename}...")
    return joblib.load(filename)


# Main function
def main():
    model_filename = "trained_models/decision_tree_model.joblib"

    # Check if we have a saved model
    if os.path.exists(model_filename):
        # Load existing model
        pruned_classifier = load_model(model_filename)

        # Load only test data for evaluation
        _, _, X_test, _, _, y_test = load_dataset()
    else:
        # Load full dataset and train new model
        X_train, X_val, X_test, y_train, y_val, y_test = load_dataset()

        # Train a decision tree classifier
        classifier = train_decision_tree(X_train, y_train, base_max_depth=3)

        # Prune the decision tree classifier
        pruned_classifier = prune_decision_tree(classifier, X_val, y_val)

        # Save the model
        save_model(pruned_classifier, model_filename)

    # Get predictions for visualization
    predictions = pruned_classifier.predict(X_test)

    # Generate UMAP visualization
    visualize_umap(X_test, predictions, "UMAP Visualization of MNIST Predictions")

    # Evaluate the decision tree classifier
    cv_score = evaluate_decision_tree_cv(pruned_classifier, X_test, y_test)
    accuracy = evaluate_decision_tree_accuracy(pruned_classifier, X_test, y_test)
    prediction_speed = measure_prediction_speed(pruned_classifier, X_test)

    # Print Results of the classifier
    print("----------- Basic Decision Tree Results -----------")
    print(f"Cross-Validation Score: {cv_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Prediction Speed: {prediction_speed:.4f} seconds")


if __name__ == "__main__":
    main()

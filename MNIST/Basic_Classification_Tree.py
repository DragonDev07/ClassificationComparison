import time

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


# Function to load the MNIST dataset and split it into training and testing sets
def load_mnist_dataset(test_size=0.2, val_size=0.2, random_state=42):
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


# Function to train a decision tree classifier
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
        estimator=base_classifier,  # `estimator` is the model to be tuned
        param_grid=param_grid,  # `param_grid` is the dictionary of hyperparameters to be tuned
        cv=5,  # `cv` is the number of folds in cross-validation
        n_jobs=-1,  # `n_jobs` is the number of parallel jobs to run (-1 means using all processors)
        verbose=3,  # `verbose` controls the verbosity
    )

    grid_search.fit(X_train, y_train)

    print(f"--> Best parameters found: {grid_search.best_params_}")
    print(f"--> Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# Function to prune a decision tree classifier (avoid overfitting)
def prune_decision_tree(classifier, X_val, y_val, random_state=42):
    # Get the cost complexity pruning path
    pruning_path = classifier.cost_complexity_pruning_path(X_val, y_val)

    # Get the alphas (controls trade-off between complexity and accuracy) and impurities
    alphas, _impurities = pruning_path.ccp_alphas, pruning_path.impurities

    # Train a decision tree classifier for each alpha
    print(f"Creating and training pruned classifiers for {len(alphas)} alphas...")
    pruned_classifiers = []
    for alpha in tqdm(alphas):
        pruned_classifier = DecisionTreeClassifier(
            random_state=random_state, ccp_alpha=alpha
        )
        pruned_classifier.fit(X_val, y_val)

        # Append the pruned classifier to the list
        pruned_classifiers.append(pruned_classifier)

    # Find the best pruned classifier tree
    print(
        f"Finding best pruned classifier out of {len(pruned_classifiers)} pruned classifiers..."
    )
    best_classifier = None
    best_cv_mean_score = 0
    for pruned_classifier in tqdm(pruned_classifiers):
        cv_score = pruned_classifier.score(X_val, y_val)

        # Update the best classifier if the current classifier is better
        if cv_score > best_cv_mean_score:
            print(
                f"--> Found better pruned classifier with cross-validation score: {cv_score}",
            )
            best_classifier = pruned_classifier
            best_cv_mean_score = cv_score

    # Return the best pruned classifier
    return best_classifier


# Function to evaluate a decision tree classifier (cross-validation)
def evaluate_decision_tree_cv(classifier, X_test, y_test, cv=5):
    print("Evaluating decision tree classifier (cross-validation)...")
    cv_scores = cross_val_score(classifier, X_test, y_test, cv=cv)

    # Return the accuracy of the classifier
    return cv_scores.mean()


# Function to evaluate a decision tree classifier (accuracy)
def evaluate_decision_tree_accuracy(classifier, X_test, y_test):
    print("Evaluating decision tree classifier (accuracy score)...")
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Return the accuracy of the classifier
    return accuracy


# Function to measure prediction speed
def measure_prediction_speed(classifier, X_test, n_trials=100):
    print("Measuring prediction speed...")
    total_time = 0

    for _ in tqdm(range(n_trials)):
        start_time = time.time()
        classifier.predict(X_test)
        end_time = time.time()

        total_time += end_time - start_time

    # Return the average prediction time
    return total_time / n_trials


def visualize_tsne(X, predictions, title="t-SNE Visualization of Predictions"):
    print("Generating & Saving t-SNE visualization...")

    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=40,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    X_tsne = tsne.fit_transform(X)

    # Create high quality scatter plot
    plt.figure(figsize=(12, 10), dpi=300)  # Increased resolution
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=predictions,
        cmap="tab10",
        alpha=0.8,  # Increased opacity
        s=100,  # Larger point size
        edgecolor="black",  # Add black edges to points
        linewidth=0.5,  # Thin edge lines
    )

    # Improve colorbar and labels
    cbar = plt.colorbar(scatter)
    cbar.set_label("Class", fontsize=12)

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot in high resolution
    plt.savefig("tsne_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()


# Main function
def main():
    # Load the MNIST dataset
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_dataset()

    # Train a decision tree classifier
    classifier = train_decision_tree(X_train, y_train, base_max_depth=3)

    # Prune the decision tree classifier
    pruned_classifier = prune_decision_tree(classifier, X_val, y_val)

    # Get predictions for visualization
    predictions = pruned_classifier.predict(X_test)

    # Generate t-SNE visualization
    visualize_tsne(X_test, predictions, "t-SNE Visualization of MNIST Predictions")

    # Evaluate the decision tree classifier
    cv_score = evaluate_decision_tree_cv(
        pruned_classifier, X_test, y_test
    )  # Cross-validation
    accuracy = evaluate_decision_tree_accuracy(
        pruned_classifier, X_test, y_test
    )  # Accuracy
    prediction_speed = measure_prediction_speed(
        pruned_classifier, X_test
    )  # Prediction Speed

    # Print Results of the classifier
    print("----------- Basic Decision Tree Results -----------")
    print(f"Cross-Validation Score: {cv_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Prediction Speed: {prediction_speed:.4f} seconds")


if __name__ == "__main__":
    main()

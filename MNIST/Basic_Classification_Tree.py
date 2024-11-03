from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


# Function to load the MNIST dataset and split it into training and testing sets
def load_mnist_dataset(test_size=0.2, random_state=42):
    print("Loading MNIST dataset...")

    mnist = fetch_openml("mnist_784", version=1)
    X, y = mnist["data"], mnist["target"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


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

    base_classifier = DecisionTreeClassifier(max_depth=base_max_depth, random_state=random_state)

    # Create GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(
        estimator=base_classifier, # `estimator` is the model to be tuned
        param_grid=param_grid, # `param_grid` is the dictionary of hyperparameters to be tuned
        cv=5, # `cv` is the number of folds in cross-validation
        n_jobs=2, # `n_jobs` is the number of parallel jobs to run (-1 means using all processors)
        verbose=3 # `verbose` controls the verbosity
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
    best_accuracy_score = 0
    for pruned_classifier in tqdm(pruned_classifiers):
        predictions = pruned_classifier.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)

        # Update the best classifier if the current classifier is better
        if accuracy > best_accuracy_score:
            print("--> Found better pruned classifier with accuracy:", accuracy)
            best_classifier = pruned_classifier
            best_accuracy_score = accuracy

    # Return the best pruned classifier
    return best_classifier


# Function to evaluate a decision tree classifier
def evaluate_decision_tree(classifier, X_test, y_test):
    print("Evaluating decision tree classifier...")
    predicitions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predicitions)

    # Return the accuracy of the classifier
    return accuracy

# Main function
def main():
    # Load the MNIST dataset
    X_train, X_test, y_train, y_test = load_mnist_dataset()

    # Train a decision tree classifier
    classifier = train_decision_tree(X_train, y_train, base_max_depth=3)

    # Prune the decision tree classifier
    pruned_classifier = prune_decision_tree(classifier, X_test, y_test)

    # Evaluate the decision tree classifier
    accuracy = evaluate_decision_tree(pruned_classifier, X_test, y_test)

    # Print Results of the classifier
    print("----------- Results -----------")
    print(f"Accuracy: {accuracy}")  # Accuracy

if __name__ == "__main__":
    main()
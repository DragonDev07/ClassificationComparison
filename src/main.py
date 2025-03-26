from models.ClassificationTree import ClassificationTree
from models.CNN import CNN
from utils.Datasets import load_mnist, load_tiny_imagenet


def main():
    print("<< === Loading Data === >>")
    print("--> Loading tiny-imagenet")
    data = load_tiny_imagenet()
    # print("--> Loading MNIST")
    # data = load_mnist()

    # Separate data into sets
    dataset_name = data["name"]
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]
    metadata = data["metadata"]

    # ---------- Classification Tree ---------- #
    print("<< === Classification Tree === >>")
    clf_tree = ClassificationTree(base_max_depth=3)
    clf_tree.dataset_name = dataset_name
    clf_tree.is_grid_search = True

    print("--> Loading Model")
    clf_tree.load()

    print("--> Running Predictions")
    predictions = clf_tree.predict(X_test)

    print("--> Generating UMAP")
    clf_tree.generate_umap(X_test, predictions)

    print("--> Evaluating Model")
    clf_tree_evaluation = clf_tree.evaluate(X_test, y_test)

    print("\n----- RESULTS -----")
    print(f"Accuracy: {clf_tree_evaluation['accuracy']:.4f}")
    print(f"Precision: {clf_tree_evaluation['precision']:.4f}")
    print(f"Recall: {clf_tree_evaluation['recall']:.4f}")
    print(f"F1 Score: {clf_tree_evaluation['f1']:.4f}")
    print(f"Prediction Speed: {clf_tree_evaluation['prediction_speed']:.4f} seconds")
    print("\nDetailed Classification Report:")
    print(clf_tree_evaluation["classification_report"])

    # --------- CNN Deep Learning ---------- #
    print("<< === CNN Deep Learning === >>")
    cnn = CNN(epochs=40, batch_size=64)
    cnn.dataset_name = dataset_name

    print("--> Loading Model")
    cnn.load()

    print("--> Running Predictions")
    cnn_predictions = cnn.predict(X_test)

    print("--> Generating UMAP")
    cnn.generate_umap(X_test, cnn_predictions)

    print("--> Evaluating Model")
    cnn_evaluation = cnn.evaluate(X_test, y_test)

    print("\n----- RESULTS -----")
    print(f"Accuracy: {cnn_evaluation['accuracy']:.4f}")
    print(f"Precision: {cnn_evaluation['precision']:.4f}")
    print(f"Recall: {cnn_evaluation['recall']:.4f}")
    print(f"F1 Score: {cnn_evaluation['f1']:.4f}")
    print(f"Prediction Speed: {cnn_evaluation['prediction_speed']:.4f} seconds")
    print("\nDetailed Classification Report:")
    print(cnn_evaluation["classification_report"])


if __name__ == "__main__":
    main()

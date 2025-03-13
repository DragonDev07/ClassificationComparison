from models.ClassificationTree import ClassificationTree
from models.CNN import CNN
from utils.Datasets import load_mnist, load_tiny_imagenet


def main():
    print("<< === Loading Data === >>")
    # print("--> Loading tiny-imagenet")
    # data = load_tiny_imagenet()
    print("--> Loading MNIST")
    data = load_mnist()

    # Separate data into sets
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    # ---------- Classification Tree ---------- #
    # print("<< === Classification Tree === >>")
    # clf_tree = ClassificationTree(base_max_depth=3)
    # clf_tree.dataset_name = "MNIST"

    # print("--> Loading Model")
    # clf_tree.load()

    # # print("--> Training Model")
    # # clf_tree.train(X_train, y_train)

    # # print("--> Pruning Model")
    # # clf_tree.prune(X_val, y_val)

    # # print("--> Saving Model")
    # # clf_tree.save()

    # print("--> Running Predictions")
    # predictions = clf_tree.predict(X_test)

    # print("--> Generating UMAP")
    # clf_tree.generate_umap(X_test, predictions)

    # print("--> Evaluating Model")
    # clf_tree_evaluation = clf_tree.evaluate(X_test, y_test)

    # print("\n----- RESULTS -----")
    # print(f"Accuracy: {clf_tree_evaluation['accuracy']}")

    # --------- CNN Deep Learning ---------- #
    print("<< === CNN Deep Learning === >>")
    cnn = CNN(epochs=20, batch_size=64)
    cnn.dataset_name = "MNIST"

    print("--> Training Model")
    cnn.train(X_train, y_train, X_val, y_val)

    print("--> Saving Model")
    cnn.save()

    print("--> Running Predictions")
    cnn_predictions = cnn.predict(X_test)

    print("--> Generating UMAP")
    cnn.generate_umap(X_test, cnn_predictions)

    print("--> Evaluating Model")
    cnn_evaluation = cnn.evaluate(X_test, y_test)

    print("\n----- RESULTS -----")
    print(f"Accuracy: {cnn_evaluation['accuracy']}")


if __name__ == "__main__":
    main()

# Classification Model Comparison: Tree-Based vs. CNN

Welcome to the LEARN.md document for the **ClassificationComparison** project. This repository stores all the code used to collect data and analyze models in my [research paper](https://docs.google.com/document/d/1qSowpSr17cUdiwWvmYBX6CnXUgwzHt94F3baRRZh8is/edit?usp=sharing).

## Getting Started

Clone the repository & install necessary dependencies:

```bash
git clone https://github.com/DragonDev07/ClassificationComparison
cd ClassificationComparison
pip install -r requirements.txt
```

Ensure your environment supports GPU acceleration if running CNN models for Tiny ImageNet.

## Project Structure

```text
ClassificationComparison
├── LEARN.md
├── out
│   ├── trained_models
│       └── <Omitted .joblib Files Storing Each Trained Model>
│   └── visualizations
│       └── <Omitted UMAP Visualization Graphs for Each Model>
├── README.md
├── requirements.txt
└── src
    ├── main.py
    ├── models
    │   ├── ClassificationTree.py
    │   └── CNN.py
    └── utils
        ├── Datasets.py
        └── ResourceMonitor.py
```

## Summary of Findings

| **Dataset**     | **Model**          | **Accuracy** | **Memory (MB)** | **Speed (s)** |
| --------------- | ------------------ | ------------ | --------------- | ------------- |
| _MNIST_         | CNN (60 Epochs)    | 99.3%        | 2585            | 0.116         |
| _MNIST_         | Tree (Grid Search) | 86.4%        | 2097            | 0.015         |
| _Tiny ImageNet_ | CNN (60 Epochs)    | 20.2%        | 9212            | 0.162         |
| _Tiny ImageNet_ | Tree (Grid Search) | 2.6%         | 6954            | 0.205         |

**Conclusion:** While CNNs outperform trees in accuracy, they require more memory and GPU support. Tree models are simpler, faster on CPUs, and better suited for lightweight tasks, though limited in complex image classification.

## Tools & Libraries Used

- Python 3.x
- `scikit-learn`, `PyTorch`, `umap-learn`
- `matplotlib`, `searborn` for visualizations

## Code Explanations

### Step 1: Preparing the Datasets

We import datasets with two functions in `Datasets.py`. The MNIST one is imported from scikitlearn with `fetch_openml`, and the Tiny ImageNet one is imported from huggingface using the `datasets` library.

### Step 2: Implementing the ClassificationTree

The Classification Tree is implemented using the `DecisionTreeClassifier` class from `sklearn.tree`. We use grid search to find the best hyperparameters from a set of predefined options and values:

```python
# grid_train() function src/models/ClassificationTree.py
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
```

### Step 3: Implementing the CNN Model

The Convolutional Neural Network (CNN) Model is implemented in the `CNN.py` file, and uses PyTorch for deep learning functionality. The model architecture consists of two main classes:

1. `CNNModel` Class: Defines the neural network architecture using PyTorch's `nn.Module`:
   - Three feature extraction blocks with increasing filter sizes (32, 64, 128)
   - Each block contains convolutional layers, batch normalization, ReLU activation, and pooling
   - Adaptive pooling to handle different input dimensions
   - Fully connected classifier layers with dropout for regularization
2. `CNN` Class: Wrapper class that provides functionality for:
   - Model initialization, training, and evaluation
   - Data preprocessing to handle different input formats
   - Prediction and visualization using UMAP
   - Resource monitoring for performance metrics
   - Model persistence (saving/loading)

## Explore Further

- Read the full paper [here](https://docs.google.com/document/d/1qSowpSr17cUdiwWvmYBX6CnXUgwzHt94F3baRRZh8is/edit?usp=sharing)
- Try tweaking:
  - Swap UMAP with t-SNE to compare dimensionality reduction techniques
  - Number of epochs for CNN
  - CNN architecture
  - Feature selection or pruning on tree models

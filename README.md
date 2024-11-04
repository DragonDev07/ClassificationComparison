# ClassificationComparison
Code to compare classification algorithms on various datasets

## Table of Contents
- [Results & Demos](#results--demos)
- [Repository Structure](#repository-structure)
    - [Datasets](#datasets)
- [Installation & Usage](#installation--usage)

## Results & Demos
<div style="text-align: center;">
    <img src="./MNIST/tsne_visualization.png" alt="t-SNE Visualization of MNIST Predictions">
    <p><em>Figure 1: t-SNE Visualization of MNIST Predictions</em></p>
</div>

## Repository Structure
- Each top-level directory specifies the dataset used for the comparison
- Within each dataset directory, there are multiple scripts that compare different classification algorithms, as well as a full script that compares all of them

### Datasets
- [MNIST](https://www.kaggle.com/c/digit-recognizer)

## Installation & Usage
1. Clone the repository
```bash
git clone https://github.com/DragonDev07/ClassificationComparison.git
```

2. Install the required packages
```bash
pip install -r requirements.txt
```

3. Run whichever script you would like to see
```bash
python3 <path_to_script>.py
```
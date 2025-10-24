# Statistical Machine Learning Project 1

## Overview
This repository contains notebooks for a binary text classification problem. Each row in the training data consists of a list of token indices (`text`), a binary label (`0` or `1`) and an `id`. The data are split into two domains. The notebooks explore a range of models from a simple logistic regression baseline to AdaBoost, SVM, MLP, LSTM and stacking ensembles.

## Files and Notebooks
- `baseline.ipynb`: TF‑IDF + logistic regression baseline.
- `data_augmentation.ipynb`: Rebalances classes by oversampling minority samples and concatenating domain 1 and domain 2 training data【614772688487699†L55-L117】.
- `SVM_diff_domain.ipynb`: Trains separate SVM models for each domain.
- `Adaboost.ipynb`: AdaBoost and AdaBoost + SVM stacking.
- `mlp_model.ipynb`: Multi‑layer perceptron in PyTorch using TF‑IDF features【57984846770243†L9-L11】.
- `lstm_model.ipynb`: LSTM using sequence embeddings.
- `meta_model.ipynb`: Stacking models trained on each domain and on the combined data【57984846770243†L12-L14】.
- `random_forest.ipynb`: Placeholder for a tree-based model.

## Dataset
Place `domain1_train_data.json` and `domain2_train_data.json` in the project root. Each has columns `text`, `label` and `id`. The notebooks load these files with `pd.read_json(..., lines=True)`【518162318634425†L51-L59】. During augmentation, the minority class is oversampled to 2 600 samples per class and the two domains are concatenated into one training set【614772688487699†L55-L117】.

## Requirements
See `requirements.txt` for a list of dependencies (pandas, numpy, scikit‑learn, gensim, matplotlib, seaborn, torch, etc.). Create a virtual environment and run `pip install -r requirements.txt`.

## Usage
Open a notebook in Jupyter and run the cells sequentially. Adjust file paths for the JSON data as needed. Experiment by tuning hyper‑parameters or adding new models. Contributions are welcome!

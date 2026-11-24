# Breast Cancer Classification

In [the notebook](main.ipynb), I compare random forest, logistic regression, and an RBF SVM on the Wisconsin Diagnostic Breast Cancer dataset. There are 569 samples, 212 malignant and 357 benign, with 30 features per sample.

## How I compared the models

I used a stratified 80/20 split and kept 114 samples for the final test. On the training set, I used five-fold cross-validation with `GridSearchCV` to choose the model, hyperparameters, and number of features. Each pipeline fits `SelectKBest` inside each fold to avoid data leakage. The logistic regression and SVM pipelines also fit `StandardScaler` there. I ran the search once for ROC AUC and once for balanced accuracy to see whether the choice of metric changed the result.

## Results

| Selected by | Model | Features | Test ROC AUC | Test balanced accuracy |
| --- | --- | ---: | ---: | ---: |
| ROC AUC | SVM | 30 | 0.99769 | 98.12% |
| Balanced accuracy | Logistic regression | 25 | 0.99537 | 96.92% |

The SVM (`C=10`, `gamma=0.01`) had the highest mean ROC AUC in cross-validation, 0.99536. On the test set it caught 41 of 42 malignant cases and correctly classified 71 of 72 benign cases.

I also checked how much the results changed across five training splits, which features mattered in permutation tests, and how the validation score changed with more training samples. With the SVM settings fixed, 20 features gave a mean CV ROC AUC of 0.99525, close to the score with all 30.

## Run the notebook

Use Python 3.12 in a virtual environment and install the dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
```

Open `main.ipynb` in VS Code with the Jupyter extension, select that environment as the kernel, and run all cells. The repeated grid searches can take a while.

The notebook loads the data through `sklearn.datasets.load_breast_cancer`. The [CSV copy](data/breast_cancer.csv) contains the same features and uses `0` for malignant and `1` for benign. Dataset by Wolberg, Mangasarian, N. Street, and W. Street (1993), [UCI Machine Learning Repository](https://doi.org/10.24432/C5DW2B), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

# Machine Failure Prediction

A data science project that uses industrial sensor readings to predict whether a machine will fail. The workflow lives in one Jupyter notebook: exploratory analysis, feature engineering, multiple classifiers, Random Forest grid search, a soft-voting ensemble, and evaluation with ROC curves, confusion matrices, and cross-validation.

This README documents how the code and data fit together so you can reproduce the analysis, interpret the metrics, and extend the pipeline safely.

## Table of Contents

[Overview](#overview) · [Repository files](#what-this-repository-contains) · [Dataset](#dataset) · [Problem](#problem-formulation) · [Methodology](#methodology) · [Models](#models-used) · [Metrics](#evaluation-metrics) · [Technical stack](#technical-stack) · [Setup & run](#installation) · [Outputs](#results-and-outputs) · [Limitations](#known-limitations) · [Extensions](#possible-extensions) · [License](#license)

---

## Overview

Predictive maintenance aims to catch failures before downtime. Here, **machine failure** is **binary classification**: from sensor and environmental readings, predict **`fail` ∈ {0, 1}`**.

The pipeline uses **EDA**, **degree-2 interaction features** on selected numerics, **one-hot encoding** of `tempMode`, **standardization**, **six sklearn/XGBoost learners**, and a **`VotingClassifier` (soft)**. Only **Random Forest** is grid-searched; other models use defaults on the prepared data.

The notebook is written as a **linear script** inside one cell: each step is labeled in comments (Step 1 through Step 14), which makes the flow easy to follow but means you should run the cell top-to-bottom without skipping intermediate assignments.

---

## What This Repository Contains

| Item | Description |
|------|-------------|
| `Projnew.ipynb` | End-to-end pipeline in **one code cell** (imports through `results_df`). |
| `data (1).csv` | Sensor table (**944 × 10** in the bundled copy). |
| `About Dataset - Machine failure prediction.txt` | Column glossary. |
| `LICENSE` | MIT License. |

---

## Dataset

| Column | Meaning |
|--------|---------|
| `footfall` | Traffic near the machine. |
| `tempMode` | Temperature mode/setting (**categorical** → dummies in code). |
| `AQ` | Air quality index. |
| `USS` | Ultrasonic proximity. |
| `CS` | Electrical current. |
| `VOC` | Volatile organic compounds. |
| `RP` | Rotational position / RPM. |
| `IP` | Input pressure. |
| `Temperature` | Operating temperature. |
| `fail` | **Target:** 1 = failure, 0 = no failure. |

Approximate class balance in the sample: **551** non-failures vs **393** failures. The split is not extreme, but **ROC-AUC**, **per-class recall**, and **stratified CV** are more informative than raw accuracy alone if the cost of missing a failure is high.

---

## Problem Formulation

**Inputs:** All engineered features (interactions, dummies, scaled). **Output:** Class or probability for `fail`.

**Goals:** Strong **ROC-AUC** and **accuracy** on a 20% hold-out (`random_state=42`), plus **confusion matrices** for error patterns and **5-fold stratified CV** on training data to estimate variance of accuracy across folds.

---

## Methodology

### Exploratory analysis

The CSV is loaded with **pandas**. The notebook prints **`describe()`** for numeric summaries, **`isnull().sum()`** for missing values, a **seaborn heatmap** of the **correlation matrix**, and a **count plot** of **`fail`** to visualize class frequency.

### Feature engineering

- **`PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)`** is applied to **`footfall`, `AQ`, `CS`, `RP`, `Temperature`**. This adds **pairwise products** among those five variables (not a full quadratic expansion of every column in the file).
- The resulting matrix is wrapped in a **DataFrame** with **`get_feature_names_out`** and **concatenated** to the original table along the column axis.
- **`tempMode`** is encoded with **`pd.get_dummies(..., drop_first=True)`** so the model sees binary columns instead of a single categorical code.

### Scaling and split

**`y = data['fail']`**, **`X`** = all other columns. **`StandardScaler`** is fit on **`X`** and the scaled array is passed to **`train_test_split`** with **`test_size=0.2`**. See [Known Limitations](#known-limitations) for a note on fit scope.

### Hyperparameters (Random Forest only)

**`GridSearchCV`** uses **5-fold CV**, **`n_jobs=-1`**, and this grid:

| Parameter | Values searched |
|-----------|-----------------|
| `n_estimators` | 50, 100, 200 |
| `max_depth` | `None`, 10, 20 |
| `min_samples_split` | 2, 5, 10 |

The best estimator **`best_rf`** is used in the ensemble and for importance plots.

### Training, ensemble, and validation

Each base model is **fit on `X_train, y_train`** and **evaluated on `X_test, y_test`**. The **soft voting** ensemble averages predicted **probabilities** from **`best_rf`**, the **in-loop trained XGBoost**, and **SVM** (which requires **`probability=True`** for `predict_proba`).

**ROC curves** overlay all base models plus the voting classifier. **Feature importances** are plotted for **`best_rf`** and **`XGBoost`** only.

**`StratifiedKFold(n_splits=5)`** with **`cross_val_score(..., scoring='accuracy')`** is run on **`X_train, y_train`** for each base model (not on the test set).

Finally, **`results_df`** tabulates **Accuracy** and **ROC AUC** for every model including **VotingClassifier**.

---

## Models Used

| Model | Role |
|-------|------|
| Random Forest | Grid-tuned; importances plotted. |
| Logistic Regression | Linear baseline. |
| SVM (RBF, `probability=True`) | Kernel baseline; used in soft voting. |
| Decision Tree | Non-ensemble tree. |
| KNN | Instance-based baseline. |
| XGBoost | Boosted trees; importances plotted. |
| Voting (soft) | `best_rf` + XGBoost + SVM. |

`GradientBoostingClassifier` is **imported but not trained** in the loop.

---

## Evaluation Metrics

- **Accuracy** — fraction of correct labels on the test split.  
- **ROC-AUC** — area under the ROC curve using **positive-class probabilities** (`predict_proba[:, 1]`).  
- **`classification_report`** — precision, recall, F1 per class.  
- **`confusion_matrix`** — true vs predicted counts for class 0 and 1.  
- **Cross-validation** — mean and standard deviation of **accuracy** over 5 stratified folds on **training** data only.

---

## Technical stack

| Layer | Libraries |
|-------|-----------|
| Data | **pandas**, **numpy** |
| ML | **scikit-learn** (models, metrics, preprocessing, CV), **xgboost** |
| Plots | **matplotlib**, **seaborn** |

---

## Installation

**Python 3.8+** and **Jupyter** are assumed. Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Using a **virtual environment** (venv, conda, or similar) is recommended so package versions stay consistent when you revisit the project.

---

## How to Run

1. Clone or download the repository.  
2. Open **`Projnew.ipynb`** in Jupyter Notebook, JupyterLab, or an editor with Jupyter support.  
3. Locate the **`pd.read_csv`** line. Replace any **absolute path** with a path relative to the notebook, for example:

   ```python
   data = pd.read_csv("data (1).csv")
   ```

   On Windows, if the notebook sits next to the CSV, the same string works when the kernel’s working directory is the project folder.

4. Execute the cell (or “Run All”). Inline plots should appear for the heatmap, class distribution, ROC comparison, and feature importances.

If **`FileNotFoundError`** appears, confirm the CSV filename matches exactly (including spaces and parentheses) or rename the file and update the string accordingly.

---

## Results and Outputs

**Console:** Dataset summary, missing-value check, per-model **accuracy**, **ROC-AUC**, **classification report**, **confusion matrix**, voting classifier metrics, cross-validation table, and **`results_df`**.

**Figures:** Correlation heatmap, **`fail`** distribution, multi-model **ROC** plot with diagonal reference, **Random Forest** and **XGBoost** feature-importance bar charts.

**Exact metric values** are not pinned in the repo; they depend on the CSV, **scikit-learn** / **xgboost** versions, and floating-point behavior. The notebook uses **commented steps** Step 1–14 in a **single cell**; “Run All” executes the full stack in order.

---

## Known Limitations

- **Hardcoded path:** The notebook may still contain a **machine-specific absolute path** for the CSV; you must edit it before running on another computer.  
- **Scaler leakage risk:** **`StandardScaler`** is **fit on the full feature matrix before splitting**. For reporting or deployment, **fit on `X_train` only** and **`transform`** both train and test (ideally inside a **`Pipeline`**).  
- **Class imbalance:** With a moderate skew toward class 0, **accuracy** can look acceptable while **recall on failures** is poor — inspect the confusion matrix and consider **class weights** or **PR curves**.  
- **Single code cell:** Debugging is easier if you split imports, EDA, training, and plotting into separate cells.  
- **Unused import:** **`GradientBoostingClassifier`** is imported but not used in the training dictionary.

---

## Possible Extensions

- Refactor with **`sklearn.pipeline.Pipeline`** and **`ColumnTransformer`** for reproducible preprocessing.  
- Add **hyperparameter search** for **XGBoost** and **SVM**, or try **LightGBM**.  
- Train **GradientBoostingClassifier** or embed **calibrated** probabilities for operations.  
- Track **average precision** or **PR-AUC** under imbalance; save the best model with **`joblib.dump`**.

---

## License

**MIT License** — see [`LICENSE`](LICENSE). Copyright (c) 2025 Shaikh Irshad Ahmed.

**Acknowledgments:** Column descriptions are summarized in **`About Dataset - Machine failure prediction.txt`**. The analysis builds on the **Python** ecosystem: **pandas**, **NumPy**, **scikit-learn**, **XGBoost**, **matplotlib**, and **seaborn**.

# Bias Analysis in Adult Census Income Classification

## Overview

This project investigates **algorithmic bias in Machine Learning models** used to predict an individual's income level based on demographic, educational, and employment-related information.

The project uses the **Adult Census Income dataset**, where the objective is to predict whether an individual's annual income is:

* `<=50K`
* `>50K`

Two classification models are trained and compared:

* **Logistic Regression**
* **Random Forest**

In addition to evaluating predictive performance, the project analyzes whether the models behave differently across demographic groups, particularly with respect to **gender and race**.

The project also includes an interpretability analysis based on model coefficients and feature importance.

---

## Project Objectives

The main objectives of the project are:

1. Prepare and preprocess the Adult Census Income dataset.
2. Train classification models to predict income level.
3. Evaluate model performance using standard classification metrics.
4. Analyze potential differences in model behavior across demographic groups.
5. Measure fairness using **Disparate Impact (DI)** and **Equal Opportunity Difference (EOD)**.
6. Analyze the most important features used by the trained models.

---

## Project Workflow

The project is organized into four main phases:

### Phase A — Data Preprocessing

The original dataset is cleaned and prepared for Machine Learning.

The preprocessing includes:

* Detection and handling of missing values represented by `?`.
* Replacement of missing categorical values with `Unknown`.
* One-Hot Encoding of categorical variables.
* Conversion of the income target into a binary variable:

  * `<=50K` → `0`
  * `>50K` → `1`
* Splitting the dataset into training and test sets using an 80/20 split.
* Stratification of the target variable to preserve the class distribution.
* Saving the processed datasets for subsequent experiments.

The original dataset contains **32,561 observations and 15 variables**. After One-Hot Encoding, the processed dataset contains **109 columns**.

### Phase B — Model Training

Two supervised classification models are trained:

* Logistic Regression
* Random Forest

The Logistic Regression model is trained using standardized features with `StandardScaler`.

Random Forest is trained directly on the processed features since tree-based models do not require feature standardization.

The trained models are saved so that they can be reused during the evaluation and analysis stages.

### Phase C — Evaluation and Fairness Analysis

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The predictions are then analyzed across demographic groups.

Two fairness metrics are used:

#### Disparate Impact (DI)

Disparate Impact compares the positive prediction rate of a protected group with that of a reference group.

A value close to `1` indicates similar positive prediction rates between the two groups, while values substantially below `1` indicate that the protected group receives positive predictions less frequently.

#### Equal Opportunity Difference (EOD)

Equal Opportunity Difference compares the True Positive Rate (TPR) of the protected group with the TPR of the reference group.

A value close to `0` indicates similar true positive rates between the two groups.

### Phase D — Interpretability

The interpretability analysis focuses on identifying the features that have the greatest influence on the models.

For Logistic Regression, the learned **model coefficients** are analyzed.

For Random Forest, the model's **feature importance** values are analyzed.

SHAP and LIME were not used because they were optional extensions of the project and were not necessary to perform the required feature-importance analysis.

---

## Results

The two models achieved similar overall performance, with an accuracy of approximately **85%**.

For Logistic Regression, the obtained accuracy was approximately **85.46%**.

The evaluation also showed that both models have more difficulty identifying the minority class (`>50K`) than the majority class (`<=50K`).

The fairness analysis revealed noticeable differences between demographic groups.

For gender, the analysis produced:

| Metric       | Female |  Male |
| ------------ | -----: | ----: |
| Accuracy     |  0.932 | 0.816 |
| Precision    |  0.767 | 0.736 |
| Recall / TPR |  0.555 | 0.620 |
| F1-score     |  0.644 | 0.673 |

The calculated **Disparate Impact was approximately 0.31**, while the **Equal Opportunity Difference was approximately -0.066**.

These results indicate that the model does not behave identically across the analyzed gender groups, despite having good overall predictive performance.

---

## Project Structure

```text
adult-income-bias-analysis/
│
├── data/
│   ├── raw/
│   │   └── adult.csv
│   │
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── models/
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_bias_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── fairness.py
│
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

The **Jupyter notebooks** are used for interactive experimentation, data inspection, visualization, and verification of the different stages of the project.

The Python modules in `src/` contain the reusable implementation of the main processing, training, evaluation, and fairness functions.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd adult-income-bias-analysis
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

The main workflow can be executed through:

```bash
python main.py
```

The Jupyter notebooks can also be used to reproduce and inspect the different stages of the analysis.

To start Jupyter Notebook:

```bash
jupyter notebook
```

Then open the notebooks located in the `notebooks/` directory.

---

## Technologies

The project was developed using:

* **Python**
* **Pandas** — data manipulation and analysis
* **NumPy** — numerical operations
* **Scikit-learn** — Machine Learning models and evaluation metrics
* **Matplotlib** — data visualization
* **Joblib** — model serialization
* **Jupyter Notebook** — interactive experimentation

---

## Dataset

The project uses the **Adult Census Income dataset**, a commonly used benchmark dataset for classification and fairness research.

The dataset contains demographic, educational, and employment-related information and is used to predict whether an individual's income exceeds $50K per year.

The dataset is used for educational and research purposes within this Machine Learning project.

---

## Fairness and Interpretability

A key aspect of this project is that high predictive performance does not necessarily imply fair behavior.

The analysis therefore combines three complementary perspectives:

```text
Predictive Performance
        |
        v
 Accuracy / Precision /
 Recall / F1-score
        |
        v
Fairness Analysis
        |
        v
 DI / EOD
        |
        v
Interpretability
        |
        v
Feature Coefficients /
Feature Importance
```

This allows the project to evaluate not only **how accurately** the models predict income, but also **how their predictions differ across demographic groups** and **which features contribute most to their decisions**.

---

## Authors

Machine Learning project developed as part of the **Master's Degree in Computer Science** at the **University of Bologna**.


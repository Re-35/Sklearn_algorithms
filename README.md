# ML Algorithms with Scikit-learn

This repository contains a collection of machine learning projects implemented using the **Scikit-learn** library. The goal is to apply core ML algorithms to real-world datasets, tune model performance, and visualize results. The projects focus on both classification and regression tasks using clean, structured Python code.

---

## 📁 Project Overview

### 1. Logistic Regression (Multi-Class Classification)

- **Description**: Implements logistic regression for multi-class classification using the popular `LogisticRegression` model from scikit-learn.
- **Key Features**:
  - Multi-class classification using `multinomial` strategie (the algorithm deprecated the multi_class parameter starting in version 1.5,
  and it will be removed in version 1.7. From then on, when there are 3 or more classes, it will always use 'multinomial' internally).
  - Regularization tuning with the inverse regularization strength `C`.
  - Preprocessing using `StandardScaler`.
  - Evaluation using accuracy scores and loss tracking.
- **Outcome**: Demonstrates the impact of regularization and solver selection on classification performance.

---

### 2. Decision Tree & Random Forest (Classification & Regression)

- **Description**: Applies a decision tree for classification and a random forest for regression using different datasets.
- **Key Features**:
  - Visualizing decision trees using `plot_tree`.
  - Entropy vs Gini for splitting criteria.
  - Regression using `RandomForestRegressor` and evaluation using plots comparing predicted vs actual values.
  - Hyperparameter tuning with `max_depth` and number of estimators.
- **Outcome**: Highlights how tree-based models adapt to different data types and problem settings.

---

### 3. Support Vector Machine (SVM)

- **Description**: Applies `SVC` (Support Vector Classifier) to visualize decision boundaries between multiple classes using selected features (e.g., BMI and Insulin).
- **Key Features**:
  - Grid search using `GridSearchCV` for tuning `C` and `gamma` in `rbf` kernel.
  - Visualization of decision boundaries using `matplotlib`.
  - Exploration of kernel effects (`linear`, `rbf`) and how margin width changes.
- **Outcome**: Illustrates SVM's strength in handling non-linear separable classes with high performance.

---

## 🧠 Skills & Techniques Applied

- Supervised Learning (Classification and Regression)
- Feature Scaling and Preprocessing
- Model Selection and Hyperparameter Tuning
- Regularization and inverse regularization techniques
- Decision boundary visualization
- Understanding bias-variance trade-off
- Evaluation metrics and graphical comparisons

---

## Datasets:
- LogisticRegression:
[Built-in penguin dataset in seaborn](https://seaborn.pydata.org/archive/0.11/tutorial/function_overview.html)
- TreeDecision and RndomForest:
[Drugs dataset in kaggle](https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees)
[Built-in california houses in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- SVM:
[Diabetes dataset in kaggle](https://www.kaggle.com/code/orjiugochukwu/ml-models-for-detecting-diabetes-with-99-accuracy/input)


## File formates:
- ipynb (jupyter notebook)
- pdf (for quick review)

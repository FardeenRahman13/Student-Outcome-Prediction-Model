# 🎓 Student Academic Outcome Prediction — CSE422 Machine Learning Project

A machine learning project that predicts whether a student will **Graduate**, remain **Enrolled**, or **Drop Out** based on academic and demographic features. Built as part of the CSE422 (Machine Learning) course, this project covers the full ML pipeline — from raw data cleaning to model evaluation and comparison.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Training](#4-model-training)
  - [5. Unsupervised Learning — KMeans Clustering](#5-unsupervised-learning--kmeans-clustering)
  - [6. Model Evaluation](#6-model-evaluation)
- [Results](#results)
- [Challenges & How They Were Handled](#challenges--how-they-were-handled)
- [Limitations & Future Improvements](#limitations--future-improvements)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)

---

## Overview

This project uses a real-world academic dataset to train and evaluate multiple ML classifiers for predicting student academic outcomes. The target variable has three classes:

| Class | Count |
|-------|-------|
| Graduate | 1979 |
| Dropout | 1273 |
| Enrolled | 719 |

The class imbalance (Graduate >> Dropout >> Enrolled) was one of the core challenges addressed during preprocessing and evaluation.

---

## Dataset

The dataset (`academic_success_dataset.csv`) contains student records with a mix of:

- **Numerical features** — e.g., admission grades, curricular units credited, GDP, inflation rate
- **Categorical features** — e.g., marital status, course, gender
- **Target** — `Graduate`, `Dropout`, or `Enrolled`

> ⚠️ Note: Two unnamed/empty columns (`Unnamed: 25`, `Unnamed: 26`) were present in the raw file and were dropped during cleaning.

---

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)

Before any modeling, the data was thoroughly explored to understand its structure:

- Identified **24 numerical** and several **categorical** features
- Checked for **class imbalance** in the target variable — classes are not equally distributed
- Visualized **distributions** using:
  - Histograms for all numerical features
  - Bar plots for categorical feature frequencies
  - **Boxplots** to identify outliers in numerical columns
- Computed **skewness** across numerical features to understand data asymmetry

### 2. Data Preprocessing

Several cleaning steps were applied to ensure data quality:

**Handling Missing Values:**
- Identified all null columns using `df.isna().any()`
- Applied **mode imputation** for categorical null values
- Applied **median imputation** for numerical null values (median is robust to outliers, unlike mean)

**Handling Outliers:**
- Boxplots were generated for every numerical feature
- **`RobustScaler`** was chosen for scaling instead of `MinMaxScaler` or `StandardScaler` — RobustScaler uses the interquartile range (IQR), making it resilient to outliers without removing them

**Encoding:**
- The `Target` column was label-encoded using `sklearn.preprocessing.LabelEncoder`:
  - `Dropout` → 0
  - `Enrolled` → 1
  - `Graduate` → 2

### 3. Feature Engineering

**Correlation Analysis:**
- Generated a full **correlation heatmap** using Pearson, Spearman, and Kendall methods
- Identified highly correlated feature pairs (correlation > 0.75)
- Dropped **`Mother's occupation`** because it had near-identical correlation with `Father's occupation` — keeping both would introduce multicollinearity
- Key findings: **Tuition fees up-to-date** and **scholarship holder** status were the most positively correlated features with graduation outcome

**Train-Test Split:**
- 70% training / 30% testing with `random_state=1` for reproducibility

### 4. Model Training

Three supervised classifiers were trained and compared:

#### K-Nearest Neighbors (KNN)
```python
KNeighborsClassifier(n_neighbors=3)
```
- Simple distance-based classifier
- Sensitive to scale, hence the use of RobustScaler beforehand

#### Decision Tree
```python
DecisionTreeClassifier()
```
- Interpretable tree-based model
- No hyperparameter tuning applied — room for improvement with `max_depth`, `min_samples_split` etc.

#### Neural Network (Keras)
```python
Sequential([
    Dense(256, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
```
- Compiled with `Adam(lr=0.01)` and `SparseCategoricalCrossentropy`
- Trained for 10 epochs with batch size 32
- Softmax output for multi-class probability

### 5. Unsupervised Learning — KMeans Clustering

In addition to supervised models, KMeans clustering was applied to explore natural groupings in the data without using the labels:

- Clustered into **3 groups** (matching the 3 known outcome classes)
- Applied **PCA (2 components)** for dimensionality reduction and visualization
- Compared clustering before and after applying `StandardScaler`
- Plotted cluster centroids on the PCA-reduced 2D space
- Variance explained by the 2 PCA components was reported on the chart title

### 6. Model Evaluation

All three classifiers were evaluated using:

- **Accuracy score** — overall and per-model bar chart comparison
- **Confusion Matrix** — per model (color-coded: Blues / Greens / Oranges)
- **Classification Report** — precision, recall, F1-score per class
- **ROC Curve** — AUC comparison for the `Graduate` class across all three models

---

## Results

| Model | Test Accuracy |
|-------|--------------|
| KNN (k=3) | ~75% |
| Decision Tree | ~72% |
| Neural Network | ~76% |

> Exact values depend on the dataset run. The Neural Network generally edged out the others, though all three were reasonably close.

The ROC curves showed that all models struggled more with the **Enrolled** class due to its smaller representation in the dataset — a known effect of class imbalance.

---

## Challenges & How They Were Handled

| Challenge | Approach |
|-----------|----------|
| Missing values in both categorical and numerical columns | Mode imputation for categorical, median for numerical |
| Outliers distorting scaling | Used `RobustScaler` instead of `MinMaxScaler` |
| Multicollinearity between features | Identified via correlation heatmap; dropped `Mother's occupation` |
| Class imbalance (Graduate >> Dropout >> Enrolled) | Acknowledged in evaluation; reflected in per-class F1 scores |
| Empty/unnamed columns in raw CSV | Detected with `isnull().sum()` and dropped before processing |
| Unsupervised clustering without labels | Dropped `Target` column before KMeans; used PCA for 2D visualization |

---

## Limitations & Future Improvements

The model performs reasonably well given the dataset, but there is meaningful room for improvement:

**Dataset Limitations:**
- The dataset has class imbalance — the `Enrolled` class is significantly underrepresented, which hurts recall for that class
- A larger, more balanced dataset would noticeably improve performance across all models
- Adding more contextual features (e.g., attendance records, mental health indicators, financial aid history) could improve predictive power

**Model Improvements:**
- Apply **SMOTE** (Synthetic Minority Oversampling Technique) to address class imbalance
- Tune Decision Tree hyperparameters (`max_depth`, `min_samples_leaf`) to reduce overfitting
- Increase Neural Network depth/epochs and add **Dropout layers** to improve generalization
- Try **Random Forest** or **XGBoost** which typically outperform single Decision Trees
- Use **GridSearchCV** or **Optuna** for systematic hyperparameter tuning
- Apply **cross-validation** instead of a single train/test split for more reliable accuracy estimates

**Evaluation Improvements:**
- Report **macro-averaged F1** as the primary metric instead of accuracy, given class imbalance
- Include a **learning curve** to diagnose underfitting vs overfitting

---

## Tech Stack

- **Python 3**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing, KNN, Decision Tree, KMeans, metrics
- `tensorflow / keras` — Neural Network
- `PCA` — dimensionality reduction for clustering visualization

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```

3. Place `academic_success_dataset.csv` in the root directory (or update the path in the notebook)

4. Open and run the notebook:
   ```bash
   jupyter notebook 422_Project.ipynb
   ```

---

## 📁 Repository Structure

```
├── 422_Project.ipynb       # Main Jupyter notebook
├── academic_success_dataset.csv  # Dataset (add manually)
└── README.md               # This file
```

---

*CSE422 — Machine Learning | BRAC University*

# Feature Engineering & Preprocessing Pipeline

This repository contains a robust, leak-free machine learning data preprocessing pipeline developed as part of the Artificial Intelligence & Machine Learning virtual internship at RabTech Academy.

## Overview

The `pipeline_task.py` script automates the cleaning and preparation of raw tabular data (using the classic Titanic dataset as a demonstration) for machine learning models. It emphasizes best practices in data science, primarily ensuring that data leakage is prevented by isolating the test set before any transformations are applied.

## Key Features

* **Data Leakage Prevention:** Executes a strict `train_test_split` prior to any imputation or scaling.
* **Automated Imputation:** Handles missing values dynamically using Median (for numerical features) and Most Frequent/Mode (for categorical features).
* **Feature Transformation:** Applies `StandardScaler` for numerical variance and `OneHotEncoder` for categorical variables.
* **Modular Pipeline Design:** Utilizes Scikit-Learn's `Pipeline` and `ColumnTransformer` for clean, reusable, and production-ready code.
* **Correlation Analysis:** Automatically generates a Seaborn heatmap to visualize feature relationships post-processing and prints the strongest correlations to the terminal.

## Prerequisites

Ensure you have Python installed on your system along with the following libraries:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib

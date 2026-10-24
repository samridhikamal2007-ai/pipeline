import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def main():
    # 1. Load the tabular dataset (Titanic dataset)
    print("Loading dataset...")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)

    # Drop features that require complex NLP or have too many unique values
    X = data.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    y = data['Survived']

    # 2. Train/Test Split (Crucial step to prevent data leakage)
    print("Splitting data to prevent leakage...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Identify numerical and categorical columns
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']

    # 4. Build Pipelines for numerical and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Fills missing with median
        ('scaler', StandardScaler())                   # Scales data
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Fills missing with mode
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # One-Hot Encoding
    ])

    # 5. Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 6. Fit and Transform
    print("Applying preprocessing pipeline...")
    # fit_transform on training data, transform only on testing data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Extract feature names for correlation analysis
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)

    # Convert processed NumPy arrays back to Pandas DataFrames
    X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    
    print("\n--- Pipeline Status ---")
    print(f"Original training shape:  {X_train.shape}")
    print(f"Processed training shape: {X_train_df.shape}")
    
    # 7. Correlation Analysis on Processed Data
    print("\nGenerating Correlation Matrix...")
    correlation_matrix = X_train_df.corr()

    # Identify highest correlations (absolute value > 0.5, excluding self-correlation)
    print("\n--- Strongest Correlations (> 0.5) ---")
    stacked_corr = correlation_matrix.abs().unstack()
    strong_corr = stacked_corr[(stacked_corr > 0.5) & (stacked_corr < 1.0)].sort_values(ascending=False).drop_duplicates()
    
    if not strong_corr.empty:
        print(strong_corr)
    else:
        print("No strong correlations found.")

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix (Post-Processing)")
    plt.tight_layout()
    plt.show() # This will pop up the visual chart

if __name__ == "__main__":
    main()
"""Linear regression models for property price prediction."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import joblib
import os
from datetime import datetime


def load_data(input_path: str = "data/processed/dvf_featured.parquet") -> pd.DataFrame:
    """Load featured dataset from Parquet file.

    Parameters:
    input_path (str): Path to the Parquet file.

    Returns:
    pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Featured data not found at: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")
    return df


def temporal_train_test_split(df: pd.DataFrame, test_year: int = 2024) -> tuple:
    """Split data temporally: train < test_year, test = test_year.

    Parameters:
    df (pd.DataFrame): Input dataframe with a 'year' column.
    test_year (int): Year to use for the test set.

    Returns:
    tuple: (train_df, test_df)
    """
    if 'year' not in df.columns:
        raise KeyError("Column 'year' required for temporal split")
    
    train_df = df[df['year'] < test_year].copy()
    test_df = df[df['year'] == test_year].copy()
    
    print(f"\nTemporal split: {len(train_df):,} train | {len(test_df):,} test")
    return train_df, test_df


def prepare_features_target(df: pd.DataFrame, target_col: str = 'log_valeur_fonciere') -> tuple:
    """Separate features and target, identify feature types.

    Parameters:
    df (pd.DataFrame): Input dataframe.
    target_col (str): Name of the target column.

    Returns:
    tuple: (X, y, numeric_features, categorical_features)
    """
    y = df[target_col].copy()
    X = df.drop(columns=['valeur_fonciere', 'log_valeur_fonciere'], errors='ignore').copy()
    
    numeric_features = X.select_dtypes(include=['int32', 'int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nFeatures: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    return X, y, numeric_features, categorical_features


def create_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Create preprocessing pipeline with imputation, scaling and encoding.

    Parameters:
    numeric_features (list): List of numeric feature names.
    categorical_features (list): List of categorical feature names.

    Returns:
    ColumnTransformer: Preprocessing pipeline.
    """
    transformers = [
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ]), categorical_features)
    ]
    
    return ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)


def train_models(X_train, y_train, preprocessor) -> dict:
    """Train and return all linear regression models.
    
    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    preprocessor (ColumnTransformer): Preprocessing pipeline.
    
    Returns:
    dict: Trained models.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=0.6),
        'Lasso Regression': Lasso(alpha=0.01, max_iter=100000)
    }
    
    fitted_models = {}
    print("\nTraining models:")
    
    for name, regressor in models.items():
        start = datetime.now()
        model = Pipeline([('preprocessor', preprocessor), ('regressor', regressor)])
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  {name}: {elapsed:.2f}s")
    
    return fitted_models


def save_all_models(fitted_models: dict, output_dir: str = "models") -> dict:
    """Save all trained models and return paths.
    
    Parameters:
    fitted_models (dict): Dictionary of trained models.
    output_dir (str): Directory to save models.

    Returns:
    dict: Paths of saved models."""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = {}
    print("\nSaving models:")
    
    for name, model in fitted_models.items():
        filename = name.replace(" ", "_").lower().replace("(", "").replace(")", "") + ".pkl"
        path = os.path.join(output_dir, filename)
        joblib.dump(model, path)
        saved_paths[name] = path
        print(f"  {name} → {path}")
    
    return saved_paths


def save_split_data(train_df, test_df, output_dir: str = "data/processed") -> None:
    """Save train/test splits for evaluation.
    
    Parameters:
    train_df (pd.DataFrame): Training dataframe.
    test_df (pd.DataFrame): Testing dataframe.
    output_dir (str): Directory to save splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(f"{output_dir}/train_split.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test_split.parquet", index=False)
    print(f"Saved splits to {output_dir}")


if __name__ == "__main__":
    print("Property price prediction - Linear models training\n")
    
    # Load and split data
    df = load_data()
    train_df, test_df = temporal_train_test_split(df)
    save_split_data(train_df, test_df)
    
    # Prepare features and preprocessing
    X_train, y_train, num_feats, cat_feats = prepare_features_target(train_df)
    preprocessor = create_preprocessing_pipeline(num_feats, cat_feats)
    
    # Train and save models
    all_models = train_models(X_train, y_train, preprocessor)
    save_all_models(all_models)
    
    print(f"\n✓ Training completed - {len(all_models)} models saved")
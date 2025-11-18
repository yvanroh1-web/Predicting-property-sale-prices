"""Linear regression models for property price prediction."""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def temporal_train_val_test_split(df: pd.DataFrame,
                                   val_year: int = 2023,
                                   test_year: int = 2024) -> tuple:
    """
    Split data into train/validation/test with temporal ordering (year col).

    Parameters:
    df (pd.DataFrame): Input dataframe
    val_year (int): Year to use for validation set (2023)
    test_year (int): Year to use for test set (2024)

    Returns:
    tuple: (train_df, val_df, test_df)

    Example: val_year=2023, test_year=2024
    → Train: <2023 (2020-2022)
    → Val: =2023
    → Test: =2024
    """
    if 'year' not in df.columns:
        raise KeyError("Column 'year' required for temporal split")

    train_df = df[df['year'] < val_year].copy()
    val_df = df[df['year'] == val_year].copy()
    test_df = df[df['year'] == test_year].copy()

    total = len(df)
    train_pct = len(train_df) / total * 100
    val_pct = len(val_df) / total * 100
    test_pct = len(test_df) / total * 100

    print("Temporal Train/Val/Test split:")
    print(f"Train (<{val_year}):  {len(train_df):>10,} samples ({train_pct:>5.2f}%)")
    print(f"Val   (={val_year}):  {len(val_df):>10,} samples ({val_pct:>5.2f}%)")
    print(f"Test  (={test_year}):  {len(test_df):>10,} samples ({test_pct:>5.2f}%)")
    print(f"Train+Val: {train_pct+val_pct:.2f}% {'✓' if train_pct+val_pct >= 80 else '✗'}")
    print(f"Test:      {test_pct:.2f}%  {'✓' if test_pct <= 20 else '✗'}")

    return train_df, val_df, test_df


def add_department_features_safe(train_df: pd.DataFrame, other_df: pd.DataFrame) -> tuple:
    """
    Add department aggregate features.

    This function:
    1. Calculates department median price (just on train data)
    2. Applies these statistics to both train and other_df (val or test)
    3. Creates relative price features

    Parameters:
    train_df (pd.DataFrame): Training dataframe
    other_df (pd.DataFrame): Validation or test dataframe

    Returns:
    tuple: (train_df_with_features, other_df_with_features)
    """
    print("Adding department features:\n")

    if 'code_departement' not in train_df.columns:
        raise KeyError("Column 'code_departement' not found")
    if 'valeur_fonciere' not in train_df.columns:
        raise KeyError("Column 'valeur_fonciere' not found")

    dept_stats_train = train_df.groupby('code_departement')['valeur_fonciere'].agg([
        ('dept_median_price', 'median')
    ]).reset_index()

    if 'dept_median_price' not in train_df.columns:
        train_df = train_df.merge(dept_stats_train, on='code_departement', how='left')
        train_df['price_vs_dept_median'] = train_df['valeur_fonciere'] / train_df['dept_median_price']
        train_df['price_vs_dept_median'] = train_df['price_vs_dept_median'].replace([np.inf, -np.inf], np.nan)

    other_df = other_df.merge(dept_stats_train, on='code_departement', how='left')
    other_df['price_vs_dept_median'] = other_df['valeur_fonciere'] / other_df['dept_median_price']
    other_df['price_vs_dept_median'] = other_df['price_vs_dept_median'].replace([np.inf, -np.inf], np.nan)

    missing_count = other_df['dept_median_price'].isna().sum()
    if missing_count > 0:
        print(f"ATTENTION! --> {missing_count} samples with unseen departments - filling with global median")
        global_median = train_df['valeur_fonciere'].median()
        other_df['dept_median_price'].fillna(global_median, inplace=True)
        other_df['price_vs_dept_median'].fillna(1.0, inplace=True)

    print("Features added: dept_median_price, price_vs_dept_median\n")

    return train_df, other_df


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
    """Train and return The 3b linear regression models.

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


def save_split_data(train_df, val_df, test_df, output_dir: str = "data/processed") -> None:
    """Save train/validation/test splits for evaluation.

    Parameters:
    train_df (pd.DataFrame): Training dataframe
    val_df (pd.DataFrame): Validation dataframe
    test_df (pd.DataFrame): Testing dataframe
    output_dir (str): Directory to save splits
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_parquet(f"{output_dir}/train_split.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/val_split.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test_split.parquet", index=False)

    print(f"\nSaved splits to {output_dir}:")
    print(f"  train_split.parquet: {len(train_df):,} samples")
    print(f"  val_split.parquet:   {len(val_df):,} samples")
    print(f"  test_split.parquet:  {len(test_df):,} samples")


if __name__ == "__main__":
    print("Property price prediction - Linear Models:")

    df = load_data()
    train_df, val_df, test_df = temporal_train_val_test_split(df, val_year=2023, test_year=2024)

    print("\nAdding department aggregate features:")
    train_df, val_df = add_department_features_safe(train_df, val_df)
    train_df, test_df = add_department_features_safe(train_df, test_df)

    save_split_data(train_df, val_df, test_df)

    X_train, y_train, num_feats, cat_feats = prepare_features_target(train_df)
    if len(X_train) > 1000000:
        print(f"\nSampling 1M rows from {len(X_train):,}...")
        idx = np.random.choice(len(X_train), size=1000000, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]

    preprocessor = create_preprocessing_pipeline(num_feats, cat_feats)

    all_models = train_models(X_train, y_train, preprocessor)
    save_all_models(all_models)

    print(f"\nTraining completed - {len(all_models)} models saved")
    print("Splits saved with validation set for evaluation")

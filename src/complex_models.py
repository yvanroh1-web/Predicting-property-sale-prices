"""Time-aware/Complex Random Forest and XGBoost models for property price prediction."""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


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


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features: year_trend, month_sin/cos, quarter.

    Parameters:
    df (pd.DataFrame): Input dataframe with 'year' and 'month' columns.

    Returns:
    pd.DataFrame: Dataframe with added temporal features."""
    df = df.copy()

    # Normalize year to [0, 1]
    year_range = df['year'].max() - df['year'].min()
    df['year_trend'] = (df['year'] - df['year'].min()) / year_range if year_range > 0 else 0

    # Cyclical encoding for seasonality
    month_rad = 2 * np.pi * df['month'] / 12
    df['month_sin'] = np.sin(month_rad)
    df['month_cos'] = np.cos(month_rad)
    df['quarter'] = (df['month'] - 1) // 3 + 1

    print("Added temporal features: year_trend, month_sin, month_cos, quarter")
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

    print("Temporal Train/Val/Test split")
    print(f"Train (<{val_year}):  {len(train_df):>10,} samples ({train_pct:>5.2f}%)")
    print(f"Val   (={val_year}):  {len(val_df):>10,} samples ({val_pct:>5.2f}%)")
    print(f"Test  (={test_year}):  {len(test_df):>10,} samples ({test_pct:>5.2f}%)")
    print(f"Train+Val: {train_pct+val_pct:.2f}% {'80%' if train_pct+val_pct >= 80 else '✗'}")
    print(f"Test:      {test_pct:.2f}%  {'20%' if test_pct <= 20 else '✗'}")

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
    tuple: (X, y, numeric_features, categorical_features)"""
    y = df[target_col].copy()
    X = df.drop(columns=['valeur_fonciere', 'log_valeur_fonciere'], errors='ignore').copy()

    # Identify and convert feature types
    numeric_features = X.select_dtypes(include=['int32', 'int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Convert category dtype to string for consistency
    for col in categorical_features:
        if X[col].dtype.name == 'category':
            X[col] = X[col].astype(str)

    print(f"\nFeatures: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    return X, y, numeric_features, categorical_features


def create_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Create preprocessing pipeline with imputation, scaling and encoding.

    Parameters:
    numeric_features (list): List of numeric feature names.
    categorical_features (list): List of categorical feature names.

    Returns:
    ColumnTransformer: Preprocessing pipeline."""
    transformers = []

    if numeric_features:
        transformers.append(('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features))

    if categorical_features:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ]), categorical_features))

    return ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)


def train_model(X_train, y_train, preprocessor, model_type: str = 'rf', sample_size: int = 1000000) -> Pipeline:
    """Train Random Forest or XGBoost model with sampling.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    preprocessor (ColumnTransformer): Preprocessing pipeline.
    model_type (str): 'r' Random Forest, 'xgb' XGBoost.
    sample_size (int): Number of samples to use if dataset is large.

    Returns:
    Pipeline: Trained model pipeline."""
    print(f"\nTraining {model_type.upper()} model...")

    # Sample if dataset is too large
    if len(X_train) > sample_size:
        print(f"  Sampling {sample_size:,} from {len(X_train):,} rows...")
        idx = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_train, y_train = X_train.iloc[idx], y_train.iloc[idx]
    else:
        print(f"  Training on full dataset: {len(X_train):,} rows")

    start = datetime.now()

    # Choose regressor
    if model_type == 'rf':
        regressor = RandomForestRegressor(
            n_estimators=50, max_depth=15, min_samples_split=20,
            min_samples_leaf=10, max_features='sqrt',
            random_state=42, n_jobs=-1, verbose=1
        )

        model = Pipeline([('preprocessor', preprocessor), ('regressor', regressor)])
        model.fit(X_train, y_train)

    else:
        regressor = XGBRegressor(
            n_estimators=100,  # reduced from 300 for speed
            learning_rate=0.05,
            max_depth=8,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            reg_alpha=2.0,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )

        # Create pipeline and fit (preprocessing happens inside pipeline)
        model = Pipeline([('preprocessor', preprocessor), ('regressor', regressor)])
        model.fit(X_train, y_train)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"  Completed in {elapsed:.2f}s ({elapsed/60:.1f} min)")

    return model


def save_model(model, output_path: str) -> None:
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved: {output_path}")


def save_split_data(train_df, val_df, test_df, output_dir: str = "data/processed") -> None:
    """Save train/validation/test splits for evaluation.

    Parameters:
    train_df (pd.DataFrame): Training dataframe
    val_df (pd.DataFrame): Validation dataframe
    test_df (pd.DataFrame): Testing dataframe
    output_dir (str): Directory to save splits
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_parquet(f"{output_dir}/train_split_timeaware.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/val_split_timeaware.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test_split_timeaware.parquet", index=False)

    print(f"\nSaved splits to {output_dir}:")
    print(f"  train_split_timeaware.parquet: {len(train_df):,} samples")
    print(f"  val_split_timeaware.parquet:   {len(val_df):,} samples")
    print(f"  test_split_timeaware.parquet:  {len(test_df):,} samples")


if __name__ == "__main__":
    print("Time-aware model training (Random Forest & XGBoost)\n")

    # Load and prepare data
    df = load_data()
    df = add_temporal_features(df)
    train_df, val_df, test_df = temporal_train_val_test_split(df, val_year=2023, test_year=2024)

    # Add department features
    print("\nAdding department aggregate features:")
    train_df, val_df = add_department_features_safe(train_df, val_df)
    train_df, test_df = add_department_features_safe(train_df, test_df)

    save_split_data(train_df, val_df, test_df)

    # Prepare features and preprocessing
    X_train, y_train, num_feats, cat_feats = prepare_features_target(train_df)
    preprocessor = create_preprocessing_pipeline(num_feats, cat_feats)

    # Train both models
    rf_model = train_model(X_train, y_train, preprocessor, model_type='rf') # Rnd forest
    save_model(rf_model, "models/time_aware/random_forest.pkl")
    xgb_model = train_model(X_train, y_train, preprocessor, model_type='xgb') # XGBoost
    save_model(xgb_model, "models/time_aware/xgboost.pkl")

    print("\nTraining completed for both models")

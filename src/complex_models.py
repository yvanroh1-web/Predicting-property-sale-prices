"""Time-aware/Complex Random Forest and XGBoost models for property price prediction."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
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


def temporal_train_test_split(df: pd.DataFrame, test_year: int = 2024) -> tuple:
    """Split data temporally: train < test_year, test = test_year.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with a 'year' column.
    test_year (int): Year to use for the test set.

    Returns:
    tuple: (train_df, test_df)"""
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
    model_type (str): 'rf' for Random Forest, 'xgb' for XGBoost.
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

    else:  # XGBoost with regularization + early stopping
        regressor = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_lambda=5.0,   # L2 regularization
            reg_alpha=2.0,    # L1 regularization
            eval_metric='rmse',
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            verbosity=1
        )

        # Split small validation set (10%) for early stopping
        val_size = int(0.1 * len(X_train))
        val_idx = np.random.choice(len(X_train), size=val_size, replace=False)
        train_mask = np.ones(len(X_train), dtype=bool)
        train_mask[val_idx] = False
        
        X_fit, y_fit = X_train.iloc[train_mask], y_train.iloc[train_mask]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Fit preprocessor and transform data
        X_fit_transformed = preprocessor.fit_transform(X_fit)
        X_val_transformed = preprocessor.transform(X_val)
        
        # Train with early stopping
        regressor.fit(
            X_fit_transformed, y_fit,
            eval_set=[(X_val_transformed, y_val)],
            verbose=True
        )
        
        model = Pipeline([('preprocessor', preprocessor), ('regressor', regressor)])

    elapsed = (datetime.now() - start).total_seconds()
    print(f"  Completed in {elapsed:.2f}s ({elapsed/60:.1f} min)")
    
    return model


def save_model(model, output_path: str) -> None:
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved: {output_path}")


def save_split_data(train_df, test_df, output_dir: str = "data/processed") -> None:
    """Save train/test splits for evaluation.
    
    Parameters:
    train_df (pd.DataFrame): Training dataframe.
    test_df (pd.DataFrame): Testing dataframe.
    output_dir (str): Directory to save splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(f"{output_dir}/train_split_timeaware.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test_split_timeaware.parquet", index=False)
    print(f"Saved splits to {output_dir}")


if __name__ == "__main__":
    print("Time-aware model training (Random Forest & XGBoost)\n")
    
    # Load and prepare data
    df = load_data()
    df = add_temporal_features(df)
    train_df, test_df = temporal_train_test_split(df)
    save_split_data(train_df, test_df)
    
    # Prepare features and preprocessing
    X_train, y_train, num_feats, cat_feats = prepare_features_target(train_df)
    preprocessor = create_preprocessing_pipeline(num_feats, cat_feats)
    
    # Train both models
    rf_model = train_model(X_train, y_train, preprocessor, model_type='rf')
    save_model(rf_model, "models/time_aware/random_forest.pkl")
    
    xgb_model = train_model(X_train, y_train, preprocessor, model_type='xgb')
    save_model(xgb_model, "models/time_aware/xgboost.pkl")
    
    print("\n✓ Training completed for both models")
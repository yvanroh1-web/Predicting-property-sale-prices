"""Evaluation script for trained models - assesses performance using MAE, RMSE, and R²."""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_splits(train_path: str = "data/processed/train_split.parquet",
                val_path: str = "data/processed/val_split.parquet",
                test_path: str = "data/processed/test_split.parquet"):
    """
    Load train, validation and test splits from Parquet files.
    
    Parameters:
    train_path (str): Path to training split.
    val_path (str): Path to validation split.
    test_path (str): Path to test split.
    
    Returns:
    tuple: train_df, val_df, test_df.
    """
    if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train/val/test splits not found. Run models.py first.")
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"Loaded train split: {len(train_df):,} rows")
    print(f"Loaded val split: {len(val_df):,} rows")
    print(f"Loaded test split: {len(test_df):,} rows")
    
    return train_df, val_df, test_df

def prepare_data(df: pd.DataFrame, target_col: str = 'log_valeur_fonciere'):
    """
    Separate features and target variable.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    target_col (str): Name of target column(to predict).
    
    Returns:
    tuple: X (features), y (target).
    """
    y = df[target_col].copy()
    X = df.drop(columns=['valeur_fonciere', 'log_valeur_fonciere'], errors='ignore').copy()
    
    return X, y

def load_models(models_dir: str = "models"):
    """
    Load all trained models from directory.
    
    Parameters:
    models_dir (str): Directory containing model files(.pkl).
    
    Returns:
    dict: Dictionary of model names and loaded models.
    """
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    models = {}
    for filename in os.listdir(models_dir):
        if filename.endswith('.pkl'):
            model_name = filename.replace('.pkl', '').replace('_', ' ').title()
            model_path = os.path.join(models_dir, filename)
            models[model_name] = joblib.load(model_path)
            print(f"Loaded: {model_name}")
    
    return models

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name: str):
    """
    Evaluate model performance on train, validation and test sets.
    
    Parameters:
    model: Trained model pipeline.
    X_train, y_train: Training features and target.
    X_val, y_val: Validation features and target.
    X_test, y_test: Test features and target.
    model_name (str): Name of the model.
    
    Returns:
    dict: Dictionary containing all evaluation metrics.
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics on log scale
    train_mae_log = mean_absolute_error(y_train, y_train_pred)
    train_rmse_log = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_mae_log = mean_absolute_error(y_val, y_val_pred)
    val_rmse_log = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    test_mae_log = mean_absolute_error(y_test, y_test_pred)
    test_rmse_log = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Convert to original price scale
    train_actual_price = np.expm1(y_train)
    train_pred_price = np.expm1(y_train_pred)
    val_actual_price = np.expm1(y_val)
    val_pred_price = np.expm1(y_val_pred)
    test_actual_price = np.expm1(y_test)
    test_pred_price = np.expm1(y_test_pred)
    
    # Calculate MAE and RMSE (use median for MAE to reduce outlier impact)
    train_mae_price = np.median(np.abs(train_actual_price - train_pred_price))
    val_mae_price = np.median(np.abs(val_actual_price - val_pred_price))
    test_mae_price = np.median(np.abs(test_actual_price - test_pred_price))
    
    train_rmse_price = np.sqrt(np.mean((train_actual_price - train_pred_price)**2))
    val_rmse_price = np.sqrt(np.mean((val_actual_price - val_pred_price)**2))
    test_rmse_price = np.sqrt(np.mean((test_actual_price - test_pred_price)**2))
    
    return {
        'Model': model_name,
        'Train MAE (log)': train_mae_log,
        'Train RMSE (log)': train_rmse_log,
        'Train R²': train_r2,
        'Val MAE (log)': val_mae_log,
        'Val RMSE (log)': val_rmse_log,
        'Val R²': val_r2,
        'Test MAE (log)': test_mae_log,
        'Test RMSE (log)': test_rmse_log,
        'Test R²': test_r2,
        'Train MAE (€)': train_mae_price,
        'Train RMSE (€)': train_rmse_price,
        'Val MAE (€)': val_mae_price,
        'Val RMSE (€)': val_rmse_price,
        'Test MAE (€)': test_mae_price,
        'Test RMSE (€)': test_rmse_price
    }

def create_comparison_table(results: list) -> pd.DataFrame:
    """
    Create and display model comparison table.
    
    Parameters:
    results (list): List of evaluation results.
    
    Returns:
    pd.DataFrame: Sorted results dataframe.
    """
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Test R²', ascending=False)
    return df_results

def plot_predictions_vs_actual(models, X_test, y_test, output_dir: str = "results/plots"):
    """
    Generate scatter plots of predicted vs actual prices.
    
    Parameters:
    models (dict): Dictionary of trained models.
    X_test: Test features.
    y_test: Test target values.
    output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        
        # Convert to original price scale
        y_test_price = np.expm1(y_test)
        y_pred_price = np.expm1(y_pred)
        
        # Sample for visualization (max 10000 points for performance)
        if len(y_test_price) > 10000:
            sample_idx = np.random.choice(len(y_test_price), 10000, replace=False)
            y_test_sample = y_test_price.iloc[sample_idx]
            y_pred_sample = y_pred_price[sample_idx]
        else:
            y_test_sample = y_test_price
            y_pred_sample = y_pred_price
        
        # Create scatter plot
        axes[idx].scatter(y_test_sample, y_pred_sample, alpha=0.3, s=10)
        axes[idx].plot([y_test_sample.min(), y_test_sample.max()], 
                       [y_test_sample.min(), y_test_sample.max()], 
                       'r--', lw=2, label='Perfect prediction')
        axes[idx].set_xlabel('Actual Price (€)')
        axes[idx].set_ylabel('Predicted Price (€)')
        axes[idx].set_title(f'{name}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'predictions_vs_actual.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

def plot_metrics_comparison(df_results: pd.DataFrame, output_dir: str = "results/plots"):
    """
    Generate bar charts comparing model metrics, fast visual return of performance.
    
    Parameters:
    df_results (pd.DataFrame): Results dataframe with metrics.
    output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = range(len(df_results))
    width = 0.25
    
    # MAE comparison
    axes[0].bar([i-width for i in x], df_results['Train MAE (€)'], width=width, alpha=0.7, label='Train', color='orange')
    axes[0].bar(x, df_results['Val MAE (€)'], width=width, alpha=0.7, label='Val', color='green')
    axes[0].bar([i+width for i in x], df_results['Test MAE (€)'], width=width, alpha=0.7, label='Test', color='steelblue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_results['Model'], rotation=45, ha='right')
    axes[0].set_ylabel('MAE (€)')
    axes[0].set_title('Mean Absolute Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RMSE comparison
    axes[1].bar([i-width for i in x], df_results['Train RMSE (€)'], width=width, alpha=0.7, label='Train', color='orange')
    axes[1].bar(x, df_results['Val RMSE (€)'], width=width, alpha=0.7, label='Val', color='green')
    axes[1].bar([i+width for i in x], df_results['Test RMSE (€)'], width=width, alpha=0.7, label='Test', color='steelblue')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_results['Model'], rotation=45, ha='right')
    axes[1].set_ylabel('RMSE (€)')
    axes[1].set_title('Root Mean Squared Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # R² comparison
    axes[2].bar([i-width for i in x], df_results['Train R²'], width=width, alpha=0.7, label='Train', color='orange')
    axes[2].bar(x, df_results['Val R²'], width=width, alpha=0.7, label='Val', color='green')
    axes[2].bar([i+width for i in x], df_results['Test R²'], width=width, alpha=0.7, label='Test', color='steelblue')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df_results['Model'], rotation=45, ha='right')
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

def save_results(df_results: pd.DataFrame, output_path: str = "results/evaluation_results.csv"):
    """
    Save evaluation results to CSV file.
    
    Parameters:
    df_results (pd.DataFrame): Results dataframe.
    output_path (str): Output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

def print_best_model(df_results: pd.DataFrame):
    """
    Display summary of best model (more accurate).
    
    Parameters:
    df_results (pd.DataFrame): Results dataframe sorted by performance.
    """
    best_model = df_results.iloc[0]
    
    print(f"\nBest Model: {best_model['Model']}")
    print(f"  Train R²: {best_model['Train R²']:.4f}")
    print(f"  Val R²: {best_model['Val R²']:.4f}")
    print(f"  Test R²: {best_model['Test R²']:.4f}")
    print(f"  Test MAE: {best_model['Test MAE (€)']:,.0f}€")
    print(f"  Test RMSE: {best_model['Test RMSE (€)']:,.0f}€")
    print(f"  Explains {best_model['Test R²']*100:.2f}% of price variance")

if __name__ == "__main__":
    # Execution of evaluation steps
    print("Model Evaluation\n")
    
    # Load train, validation and test data
    train_df, val_df, test_df = load_splits()
    X_train, y_train = prepare_data(train_df)
    X_val, y_val = prepare_data(val_df)
    X_test, y_test = prepare_data(test_df)
    
    # Sample data also in eval for Github Actions
    print("\nSampling data for evaluation:")
    if len(X_train) > 100000:
        idx = np.random.choice(len(X_train), size=100000, replace=False)
        X_train, y_train = X_train.iloc[idx], y_train.iloc[idx]
        print(f"  Train: sampled 100k from original size")
    
    if len(X_val) > 50000:
        idx = np.random.choice(len(X_val), size=50000, replace=False)
        X_val, y_val = X_val.iloc[idx], y_val.iloc[idx]
        print(f"  Val: sampled 50k from original size")
    
    if len(X_test) > 50000:
        idx = np.random.choice(len(X_test), size=50000, replace=False)
        X_test, y_test = X_test.iloc[idx], y_test.iloc[idx]
        print(f"  Test: sampled 50k from original size")
    
    # Load trained models
    print("\nLoading models:")
    models = load_models()
    
    # Evaluate all models
    print("\nEvaluating models:")
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, name)
        results.append(result)
        print(f"  {name}: Val R²={result['Val R²']:.4f}, Test R²={result['Test R²']:.4f}, Test MAE={result['Test MAE (€)']:,.0f}€")
    
    # Create comparison table
    df_results = create_comparison_table(results)
    
    # Print best model summary
    print_best_model(df_results)
    
    print("Generating visualizations:")
    plot_predictions_vs_actual(models, X_test, y_test)
    plot_metrics_comparison(df_results)
    
    # Save results to CSV
    save_results(df_results)
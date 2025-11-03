"""Evaluation script for time-aware models, evaluation metrics MAE, RMSE, and R²."""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from evaluation_linear import plot_metrics_comparison
import matplotlib.pyplot as plt
import seaborn as sns

def load_splits(train_path: str = "data/processed/train_split_timeaware.parquet", 
                test_path: str = "data/processed/test_split_timeaware.parquet"):
    """
    Load train and test splits from Parquet files.
    
    Parameters:
    train_path (str): Path to training split.
    test_path (str): Path to test split.
    
    Returns:
    tuple: train_df, test_df.
    """
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train/test splits not found. Run models.py first.")
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"Loaded train split: {len(train_df):,} rows")
    print(f"Loaded test split: {len(test_df):,} rows")
    
    return train_df, test_df

def prepare_data(df: pd.DataFrame, target_col: str = 'log_valeur_fonciere'):
    """
    Separate features and target variable.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    target_col (str): Name of target column.
    
    Returns:
    tuple: X (features), y (target).
    """
    y = df[target_col].copy()
    X = df.drop(columns=['valeur_fonciere', 'log_valeur_fonciere'], errors='ignore').copy()
    
    return X, y

def load_model(model_path: str = "models/time_aware/random_forest.pkl"):
    """
    Load trained Random Forest model.
    
    Parameters:
    model_path (str): Path to model file.
    
    Returns:
    model: Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Loaded model: {model_path}")
    
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str = "Random Forest"):
    """
    Evaluate model performance on train and test sets.
    
    Parameters:
    model: Trained model pipeline.
    X_train, y_train: Training features and target.
    X_test, y_test: Test features and target.
    model_name (str): Name of the model.
    
    Returns:
    dict: Dictionary containing all evaluation metrics.
    """
    print(f"\nEvaluating {model_name}...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics on log scale
    train_mae_log = mean_absolute_error(y_train, y_train_pred)
    train_rmse_log = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae_log = mean_absolute_error(y_test, y_test_pred)
    test_rmse_log = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Convert to original price scale
    train_actual_price = np.expm1(y_train)
    train_pred_price = np.expm1(y_train_pred)
    test_actual_price = np.expm1(y_test)
    test_pred_price = np.expm1(y_test_pred)
    
    # Calculate MAE and RMSE in euros (use median for MAE to reduce outlier impact)
    train_mae_price = np.median(np.abs(train_actual_price - train_pred_price))
    test_mae_price = np.median(np.abs(test_actual_price - test_pred_price))
    
    # Clip extreme values for RMSE to prevent overflow
    max_price = 10_000_000
    train_actual_clipped = np.clip(train_actual_price, 0, max_price)
    train_pred_clipped = np.clip(train_pred_price, 0, max_price)
    test_actual_clipped = np.clip(test_actual_price, 0, max_price)
    test_pred_clipped = np.clip(test_pred_price, 0, max_price)
    
    train_rmse_price = np.sqrt(np.mean((train_actual_clipped - train_pred_clipped)**2))
    test_rmse_price = np.sqrt(np.mean((test_actual_clipped - test_pred_clipped)**2))
    
    # Print results
    print(f"\n{model_name} Performance:")
    print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"  Train MAE: {train_mae_price:,.0f}€ | Test MAE: {test_mae_price:,.0f}€")
    print(f"  Train RMSE: {train_rmse_price:,.0f}€ | Test RMSE: {test_rmse_price:,.0f}€")
    
    return {
        'Model': model_name,
        'Train MAE (log)': train_mae_log,
        'Train RMSE (log)': train_rmse_log,
        'Train R²': train_r2,
        'Test MAE (log)': test_mae_log,
        'Test RMSE (log)': test_rmse_log,
        'Test R²': test_r2,
        'Train MAE (€)': train_mae_price,
        'Train RMSE (€)': train_rmse_price,
        'Test MAE (€)': test_mae_price,
        'Test RMSE (€)': test_rmse_price
    }

def plot_predictions_vs_actual(model, X_test, y_test, model_name: str = "Random Forest", 
                               output_dir: str = "results/plots_timeaware"):
    """
    Generate scatter plot of predicted vs actual prices.
    
    Parameters:
    model: Trained model.
    X_test: Test features.
    y_test: Test target values.
    model_name (str): Name of the model.
    output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
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
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_sample, y_pred_sample, alpha=0.3, s=10)
    plt.plot([y_test_sample.min(), y_test_sample.max()], 
             [y_test_sample.min(), y_test_sample.max()], 
             'r--', lw=2, label='Perfect prediction')
    plt.xlabel('Actual Price (€)', fontsize=12)
    plt.ylabel('Predicted Price (€)', fontsize=12)
    plt.title(f'{model_name} - Predicted vs Actual Prices', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'predictions_vs_actual_timeaware.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

def plot_residuals(model, X_test, y_test, model_name: str = "Random Forest",
                  output_dir: str = "results/plots_timeaware"):
    """
    Generate residual distribution histogram.
    
    Parameters:
    model: Trained model.
    X_test: Test features.
    y_test: Test target values.
    model_name (str): Name of the model.
    output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Residuals (log scale)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{model_name} - Residual Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'residuals_distribution_timeaware.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

def compare_with_baseline(timeaware_results: dict, baseline_path: str = "results/evaluation_results.csv"):
    """
    Compare time-aware model with baseline linear models.
    
    Parameters:
    timeaware_results (dict): Time-aware model results.
    baseline_path (str): Path to baseline results CSV.
    """
    if not os.path.exists(baseline_path):
        print(f"\nBaseline results not found at {baseline_path}")
        print("Run evaluation.py for linear models first to generate comparison.")
        return
    
    baseline_df = pd.read_csv(baseline_path)
    
    print("\n" + "=" * 100)
    print("COMPARISON: TIME-AWARE vs BASELINE MODELS")
    print("=" * 100)
    
    print("\nBaseline Models (Linear, Ridge, Lasso):")
    print(baseline_df[['Model', 'Test R²', 'Test MAE (€)', 'Test RMSE (€)']].to_string(index=False))
    
    print(f"\nTime-Aware Model ({timeaware_results['Model']}):")
    print(f"  Test R²: {timeaware_results['Test R²']:.4f}")
    print(f"  Test MAE: {timeaware_results['Test MAE (€)']:,.0f}€")
    print(f"  Test RMSE: {timeaware_results['Test RMSE (€)']:,.0f}€")
    
    # Calculate improvement
    best_baseline_r2 = baseline_df['Test R²'].max()
    best_baseline_mae = baseline_df['Test MAE (€)'].min()
    
    r2_improvement = ((timeaware_results['Test R²'] - best_baseline_r2) / best_baseline_r2) * 100
    mae_improvement = ((best_baseline_mae - timeaware_results['Test MAE (€)']) / best_baseline_mae) * 100
    
    print(f"\nImprovement over best baseline:")
    print(f"  R² change: {r2_improvement:+.2f}%")
    print(f"  MAE change: {mae_improvement:+.2f}%")
    print("=" * 100)

def save_results(results: dict, output_path: str = "results/evaluation_timeaware.csv"):
    """
    Save evaluation results to CSV file.
    
    Parameters:
    results (dict): Results dictionary.
    output_path (str): Output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results = pd.DataFrame([results])
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    print("Time-Aware Model Evaluation\n")
    
    # Load train and test data
    train_df, test_df = load_splits()
    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)
    
    # Load trained models 
    models = {}
    for model_file in ["random_forest.pkl", "xgboost.pkl"]:
        path = f"models/time_aware/{model_file}"
        if os.path.exists(path):
            model_name = model_file.replace(".pkl", "").replace("_", " ").title()
            models[model_name] = joblib.load(path)
            print(f"Loaded: {model_name}")

    results = []
    for name, model in models.items():
        res = evaluate_model(model, X_train, y_train, X_test, y_test, model_name=name)
        results.append(res)
        print()

        # Plot individual visualizations
        plot_predictions_vs_actual(model, X_test, y_test, model_name=name)
        plot_residuals(model, X_test, y_test, model_name=name)
    
    # Create and save metrics comparison (if multiple models)
    if len(results) > 1:
        df_results = pd.DataFrame(results).sort_values('Test R²', ascending=False)
        
        # Plotting style like for linear models
        plot_metrics_comparison(df_results, output_dir="results/plots_timeaware")
        
        df_results.to_csv("results/evaluation_timeaware.csv", index=False)
        print("Saved aggregated results to: results/evaluation_timeaware.csv")
    else:
        # Save single model result
        save_results(results[0])
        compare_with_baseline(results[0])
    
    print("\nEvaluation completed")
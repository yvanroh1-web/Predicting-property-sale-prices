"""Feature engineering script to create predictive features from the preprocessed dataset."""

import pandas as pd
import numpy as np
import os

def load_data(input_path: str = "data/processed/dvf_processed.parquet") -> pd.DataFrame:
    """
    Load preprocessed Parquet data file.

    Parameters:
    input_path (str): Path to the preprocessed Parquet file.

    Returns:
    pd.DataFrame: Loaded DataFrame from Parquet file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Parquet file not found at: {input_path}")
    
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")
    
    return df

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the 'date_mutation' column.

    Features added:
    - 'year': Transaction year
    - 'month': Transaction month (seasonality component)

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: Feature-enriched DataFrame.
    """
    if 'date_mutation' not in df.columns:
        raise KeyError("Column 'date_mutation' is missing from DataFrame")

    # Ensure date_mutation is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date_mutation']):
        df['date_mutation'] = pd.to_datetime(df['date_mutation'], errors='coerce')

    df['year'] = df['date_mutation'].dt.year
    df['month'] = df['date_mutation'].dt.month

    print(f"Added temporal features: year, month")

    return df

def convert_price_to_logarithmic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply logarithmic transformation to sale prices to reduce skewness,
    while preserving the original numeric price.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with added column 'log_valeur_fonciere'.
    """
    if 'valeur_fonciere' not in df.columns:
        raise KeyError("Column 'valeur_fonciere' is missing from DataFrame")

    df['log_valeur_fonciere'] = np.log1p(df['valeur_fonciere'])
    
    print(f"Created log-transformed price feature")

    return df

def add_department_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add aggregate feature department-level median price.
    
    Features added:
    - 'dept_median_price': Median price per department (robust to outliers)
    - 'price_vs_dept_median': Ratio of property price to department median
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with aggregate features.
    """
    if 'code_departement' not in df.columns:
        raise KeyError("Column 'code_departement' is missing from DataFrame")
    if 'valeur_fonciere' not in df.columns:
        raise KeyError("Column 'valeur_fonciere' is missing from DataFrame")
    
    # Calculate department-level median price
    dept_stats = df.groupby('code_departement')['valeur_fonciere'].agg([
        ('dept_median_price', 'median')
    ]).reset_index()
    
    # Merge aggregates back to original dataframe
    df = df.merge(dept_stats, on='code_departement', how='left')
    
    # Create relative price feature (how much respect to department median)
    df['price_vs_dept_median'] = df['valeur_fonciere'] / df['dept_median_price']
    
    # Handle issues division by zero or NaN
    df['price_vs_dept_median'] = df['price_vs_dept_median'].replace([np.inf, -np.inf], np.nan)
    
    print(f"Added department aggregate features:")
    print(f"  - dept_median_price: Median price per department")
    print(f"  - price_vs_dept_median: Ratio to department median")
    
    return df

def handle_outliers_percentile(df: pd.DataFrame, lower: float = 0.001, upper: float = 0.999) -> pd.DataFrame:
    """
    Remove outliers based on price percentiles only.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    lower (float): Lower percentile threshold (default: 0.001 = 0.1%)
    upper (float): Upper percentile threshold (default: 0.999 = 99.9%)
    
    Returns:
    pd.DataFrame: Filtered DataFrame without extreme outliers.
    """
    initial_count = len(df)
    
    # Define bounds for price only
    price_lower = df['valeur_fonciere'].quantile(lower)
    price_upper = df['valeur_fonciere'].quantile(upper)
    
    # Filter based on price only
    df = df[(df['valeur_fonciere'] >= price_lower) & 
            (df['valeur_fonciere'] <= price_upper)].copy()
    
    removed = initial_count - len(df)
    percentage = (removed / initial_count * 100)
    print(f"Outliers removed (percentile method): {removed:,} rows ({percentage:.2f}%)")
    print(f"  Price range kept: {price_lower:,.0f}€ to {price_upper:,.0f}€")
    
    return df

def select_modeling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only relevant features for modeling, drop IDs and redundant columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with selected features only.
    """
    # Columns to keep for modeling (NO price_per_m2!)
    keep_columns = [
        # Target variables
        'valeur_fonciere',
        'log_valeur_fonciere',
        
        # Property characteristics
        'surface_reelle_bati',
        'nombre_pieces_principales',
        'surface_terrain',
        'nombre_lots',
        
        # Categorical features
        'type_local',
        'nature_mutation',
        'code_departement',
        
        # Geographic features
        'longitude',
        'latitude',
        
        # Temporal features
        'year',
        'month',
        
        # Department aggregate features
        'dept_median_price',
        'price_vs_dept_median',
    ]
    
    # Keep only available columns
    available_columns = [col for col in keep_columns if col in df.columns]
    missing_columns = [col for col in keep_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    df = df[available_columns].copy()
    
    print(f"Selected {len(available_columns)} features for modeling")
    
    return df

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute complete feature-engineering pipeline.

    Steps:
    1. Add time-based predictive features (year, month)
    2. Add departement aggregate features (median price)
    3. Log-transform prices (stabilize skew)
    4. Remove statistical outliers (percentile method)
    5. Select relevant features for modeling
    6. Drop rows with missing values in critical features

    Returns:
    pd.DataFrame: Final processed DataFrame ready for modeling.
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    print(f"Starting with {len(df):,} rows\n")

    # Step 1-3: Feature creation
    df = add_datetime_features(df)
    df = add_department_aggregates(df)
    df = convert_price_to_logarithmic(df)

    # Step 4: Remove outliers using percentile method
    print()
    df = handle_outliers_percentile(df, lower=0.001, upper=0.999)
    
    # Step 5: Select modeling features
    print()
    df = select_modeling_features(df)

    # Step 6: Drop rows with missing critical features
    print()
    significant_features = ['log_valeur_fonciere']
    before_drop = len(df)
    df = df.dropna(subset=significant_features)
    
    dropped = before_drop - len(df)
    percentage = (dropped / before_drop * 100) if before_drop > 0 else 0
    print(f"Removed {dropped:,} rows ({percentage:.1f}%) with missing critical features")
    
    print("\n" + "=" * 80)
    print(f"Feature Engineering completed with {len(df):,} rows")
    print("=" * 80 + "\n")

    return df

def save_featured_data(df: pd.DataFrame, output_path: str = "data/processed/dvf_featured.parquet") -> None:
    """
    Save the featured DataFrame to a Parquet file.

    Parameters:
    df (pd.DataFrame): The featured dataset.
    output_path (str): Destination path for the Parquet file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Featured data saved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    # Execution of feature-engineering steps
    df = load_data()
    df = process_features(df)
    save_featured_data(df)
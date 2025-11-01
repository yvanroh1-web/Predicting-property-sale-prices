"Preprocessing script charged for cleaning and preparing the dataset."

import pandas as pd
import numpy as np
import os

def load_data(folder) -> pd.DataFrame:
    """
    Load CSV data files from the specified folder into a collection of DataFrames.

    Parameters:
    folder (str): The path to the folder containing the CSV files.

    Returns:
    df_dvf: A single DataFrame cointaining the merge of all CSV files.
    """
    data_frames = []
    # Read all csv file in the folder
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder, filename)
            df = pd.read_csv(file_path, low_memory=False)
            data_frames.append(df)

    if not data_frames:
        raise ValueError("No CSV files found in the specified folder.") 
    
    return pd.concat(data_frames, ignore_index=True)

def remove_empty_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove columns with more than `threshold` % missing values.
    Parameters: 
    df (pd.DataFrame): Input dataframe.
    Default: 
    threshold is 90%.
    Returns:
    pd.DataFrame: Dataframe with columns having less than 90% of missing values.
    """
    return df.loc[:, df.isnull().mean() < threshold]

def filter_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid transactions: non-positive price and missing coordinates.
    Parameters:
    df (pd.DataFrame): Input dataframe.
    Returns:
    pd.DataFrame: Filtered dataframe.
    """
    df = df[df['valeur_fonciere'] > 0] # Remove non-positive prices
    df = df.dropna(subset=['longitude', 'latitude']) # Remove missing coordinates
    df['date_mutation'] = pd.to_datetime(df['date_mutation'], errors='coerce') # Convert the date_mutation column in datetime format
    df = df[df['date_mutation'].notna()] # Remove rows with invalid dates
    return df

def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize data types for problematic columns before saving to Parquet.
    Converts columns that should be strings but contain mixed types.
    """
    # Converti code_commune e altre colonne simili in stringhe
    string_columns = ['code_commune', 'code_departement', 'code_postal']
    
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', np.nan)
    
    return df

def save_clean_data(df: pd.DataFrame, output_path: str = "data/processed/dvf_processed.parquet") -> None:
    """
    Save the cleaned dataset to a Parquet file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved cleaned dataset → {output_path}")

if __name__ == "__main__":
    # Execution of preprocessing steps
    raw_folder = "data/raw"
    df = load_data(raw_folder)
    initial_rows = len(df)
    print(f"Total rows before preprocessing process: {initial_rows}")

    df = remove_empty_columns(df)
    df = filter_transactions(df)
    df = standardize_dtypes(df)
    print(f"Total of rows removed after preprocessing process: {initial_rows - len(df)}")

    save_clean_data(df)
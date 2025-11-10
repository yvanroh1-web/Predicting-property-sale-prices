"""Preprocessing script for cleaning and preparing the DVF dataset (2020-2024)."""

import pandas as pd
import numpy as np
import os
import requests
import gzip
from tqdm import tqdm

def download_dvf_data(output_folder: str = "data/raw", years: list = [2020, 2021, 2022, 2023, 2024]) -> None:
    """
    Download DVF dataset for specified years (2020-2024) from data.gouv.fr.
    
    Parameters:
    output_folder (str): Folder where to save the downloaded CSV files
    years (list): List of years to download (default: 2020-2024)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print("Downloading DVF dataset:")
    print(f"Years to download: {years}")
    print(f"Destination folder: {output_folder}\n")
    
    base_url = "https://files.data.gouv.fr/geo-dvf/latest/csv"
    
    for year in years:
        url = f"{base_url}/{year}/full.csv.gz"
        output_path = os.path.join(output_folder, f"dvf_{year}.csv")
        
        # Skip if already downloaded
        if os.path.exists(output_path):
            print(f"  {year}: Already downloaded, skipping")
            continue
        
        try:
            print(f"  {year}: Downloading from {url}...")
            
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            compressed_data = b""
            
            # Download with progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"  {year}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    compressed_data += chunk
                    pbar.update(len(chunk))
            
            # Decompress
            print(f"  {year}: Decompressing...")
            decompressed_data = gzip.decompress(compressed_data)
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(decompressed_data)
            
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  {year}:Saved ({file_size_mb:.1f} MB)\n")
            
        except requests.exceptions.RequestException as e:
            print(f"  {year}:Error downloading: {e}\n")
            continue
    
    print("Download completed!\n")

def load_data(folder: str) -> pd.DataFrame:
    """
    Load all CSV files from the specified folder and merge them.
    
    Parameters:
    folder (str): Path to the folder containing the CSV files
    
    Returns:
    pd.DataFrame: Merged DataFrame
    """
    print("Loading and merge splitted dataset")
    
    data_frames = []
    
    # Read all CSV files in the folder
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.csv') and filename.startswith('dvf_'):
            file_path = os.path.join(folder, filename)
            print(f"Loading {filename}...")
            df = pd.read_csv(file_path, low_memory=False)
            data_frames.append(df)
            print(f"  Loaded {len(df):,} rows")
    
    if not data_frames:
        raise ValueError("No CSV files found in the specified folder.")
    
    print(f"\nMerging {len(data_frames)} files...")
    df_merged = pd.concat(data_frames, ignore_index=True)
    
    print(f"Total rows: {len(df_merged):,}")
    print(f"Columns: {len(df_merged.columns)}")
    print(f"Memory usage: {df_merged.memory_usage(deep=True).sum() / (1024**3):.2f} GB\n")
    
    return df_merged

def remove_empty_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove columns with more than `threshold` % missing values.
    
    Parameters: 
    df (pd.DataFrame): Input dataframe
    threshold (float): Maximum allowed missing percentage (default: 0.9 = 90%)
    
    Returns:
    pd.DataFrame: Dataframe with columns having less than 90% missing values
    """
    print("Removing columns with excessive missing values:")
    
    missing_pct = df.isnull().mean()
    cols_to_remove = missing_pct[missing_pct >= threshold].index.tolist()
    
    print(f"Threshold: {threshold*100:.0f}% missing values")
    print(f"Columns before: {len(df.columns)}")
    print(f"Columns to remove: {len(cols_to_remove)}")
    
    if cols_to_remove:
        print("\nRemoved columns:")
        for col in cols_to_remove:
            print(f"  - {col}: {missing_pct[col]*100:.1f}% missing")
    
    df_clean = df.loc[:, df.isnull().mean() < threshold]
    
    print(f"\nColumns after: {len(df_clean.columns)}\n")
    
    return df_clean

def filter_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid transactions: non-positive prices and missing coordinates.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Filtered dataframe
    """
    print("Filtering invalid transactions: ")
    
    initial_count = len(df)
    
    df = df[df['valeur_fonciere'] > 0]
    after_price = len(df)
    print(f"Removed non-positive prices:  {initial_count - after_price:>10,} rows")
    
    df = df.dropna(subset=['longitude', 'latitude'])
    after_coords = len(df)
    print(f"Removed missing coordinates:  {after_price - after_coords:>10,} rows")
    
    df = df[df['date_mutation'].notna()]
    after_dates = len(df)
    print(f"Removed invalid dates:        {after_coords - after_dates:>10,} rows")
    
    print(f"\nTotal rows removed:           {initial_count - after_dates:>10,}")
    print(f"Remaining rows:               {after_dates:>10,}")
    print(f"Percentage kept:              {after_dates/initial_count*100:>10.1f}%\n")
    
    return df

def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize data types for problematic columns before saving to Parquet.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Dataframe with standardized dtypes
    """
    print("Standardizing data types")
    
    string_columns = ['code_commune', 'code_departement', 'code_postal']
    
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', np.nan)
            print(f"Converted {col} to string")
    
    return df

def save_clean_data(df: pd.DataFrame, output_path: str = "data/processed/dvf_processed.parquet") -> None:
    """
    Save the cleaned dataset to a Parquet file.
    
    Parameters:
    df (pd.DataFrame): Cleaned dataframe
    output_path (str): Path where to save the Parquet file
    """
    print("Saving cleaned dataset")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving to: {output_path}")
    df.to_parquet(output_path, index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Final dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"File size: {file_size_mb:.1f} MB\n")

if __name__ == "__main__":
    print("DVF dataset (2020-2024) preprocessing:")
    print("\nData source: https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/")

    IS_CI = os.getenv("GITHUB_ACTIONS") == "true"
    raw_folder = "data/raw"

    if IS_CI:
        print("Detected CI environment – using local raw data from repository")
        if not os.path.exists(raw_folder) or not any(f.endswith(".csv") for f in os.listdir(raw_folder)):
            raise FileNotFoundError("No CSV files found in data/raw/. Please include DVF raw files in the repository.")
        df = load_data(raw_folder)
    else:
        download_dvf_data(raw_folder)
        df = load_data(raw_folder)
    
    df = remove_empty_columns(df)
    df = filter_transactions(df)
    df = standardize_dtypes(df)
    
    save_clean_data(df)
    
    print("Preprocessing completed")
    print(f"Final rows: {len(df):,}")
    print(f"Final columns: {len(df.columns)}")

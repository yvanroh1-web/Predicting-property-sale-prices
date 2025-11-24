# Predicting Property Sale Prices in France

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![CI/CD Pipeline](https://github.com/yvanroh1-web/Predicting-property-sale-prices/actions/workflows/pipeline.yml/badge.svg)](https://github.com/yvanroh1-web/Predicting-property-sale-prices/actions/workflows/pipeline.yml)
[![Pylint](https://github.com/yvanroh1-web/Predicting-property-sale-prices/actions/workflows/pylint.yml/badge.svg)](https://github.com/yvanroh1-web/Predicting-property-sale-prices/actions/workflows/pylint.yml)

A comprehensive machine learning pipeline to predict residential property prices in France using the Demandes de Valeurs Foncières (DVF) dataset, comprising 18.1 million transactions recorded between 2020 and 2024. This project implements a complete data science workflow with temporal validation, ensemble learning methods, and automated CI/CD deployment.

## Project Overview

Property valuation is a critical component of real estate markets, yet traditional models often fail to capture complex non-linear relationships between property characteristics and sale prices. <br>
This project develops an end-to-end predictive system that substantially outperforms linear approaches through tree-based ensemble methods.

**Key Features:**
- **18.1 million property transactions** from French open data (2020-2024)
- **Five ML models:** Linear Regression, Ridge, Lasso, Random Forest, XGBoost
- **Advanced feature engineering** with 24 predictive variables across temporal, geographic, and property attributes
- **Reproducible pipeline** with automated CI/CD via GitHub Actions

## Key Results

| Model | Test R² | Test MAE (€) | Test RMSE (€) |
|-------|---------|--------------|---------------|
| **Random Forest** - <p style="font-weight: bold; color: red;">BEST</p> | **0.77** | **36,878** | **4,883,789** |
| Linear Regression | 0.26 | 74,460 | 5,198,559 |
| Ridge Regression | 0.26 | 74,510 | 5,198,745 |
| Lasso Regression | 0.24 | 70,440 | 5,201,234 |
| XGBoost* | 0.99* | 1,890* | 2,460,893* |

**Random Forest achieved 197% improvement over linear models**, demonstrating that ensemble tree-based methods substantially outperform traditional regression approaches for large-scale property price prediction.

## Technical Report

**[Complete report (PDF)](docs/report.pdf)**

## Quick Start

### Prerequisites

- **Python 3.10**
- **Git** with Git LFS (for large dataset files)
- **pip** (Python package manager)

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/yvanroh1-web/Predicting-property-sale-prices.git
cd Predicting-property-sale-prices
```

#### 2. Create virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Pipeline Architecture

The ML pipeline consists of six scripts that execute sequentially:

```
Data Flow:
DVF Raw CSV Files → Preprocessing → Feature Engineering → Model Training → Evaluation → Results
```

### Step 1: Data Preprocessing
```bash
python src/preprocessing.py
```

**Purpose:** Merge and clean raw DVF CSV files
- Concatenate all property transaction files (2020-2024)
- Remove columns with >90% missing values
- Filter invalid transactions (non-positive prices, missing coordinates)
- Standardize date and regional identifiers

**Output:** `data/processed/dvf_processed.parquet` 

**Note:** This step is not included in CI/CD automation due to Git LFS monthly bandwidth limits for large CSV files.

### Step 2: Feature Engineering
```bash
python src/feature_engineering.py
```

**Purpose:** Transform raw administrative records into predictive variables

**Feature Creation (24 variables across 4 categories):**
- **Temporal Features:** Year extraction, cyclical month encoding (sin/cos), quarterly indicators
- **Geographic Features:** Department median price, relative price ratio, spatial coordinates
- **Property Attributes:** Price per m², property type (one-hot encoded), surface measurements, room count
- **Target Transformation:** Log transformation to reduce skewness (2.8 → 0.3)

**Model-Specific Selection:**
- **Linear Models (14 features):** Numeric variables + one-hot encoded categoricals, StandardScaler normalization
- **Tree Models (18 features):** All numeric variables including derived ratios + label-encoded categoricals

**Output:** `data/processed/dvf_featured.parquet`

### Step 3: Model Training - Linear Models
```bash
python src/linear_models.py
```

**Purpose:** Train baseline regression models
- Linear Regression
- Ridge Regression  
- Lasso Regression

**Output:** `models/linear_*.pkl`

### Step 4: Model Training - Complex Models
```bash
python src/complex_models.py
```

**Purpose:** Train ensemble tree-based models
- Random Forest
- XGBoost

**Optimization:** Stratified sampling (1-5M rows) for computational efficiency

**Output:** `models/random_forest.pkl`, `models/xgboost.pkl`

### Step 5: Evaluation - Linear Models
```bash
python src/evaluation_linear.py
```

**Purpose:** Evaluate linear model performance on test set (2024)

**Metrics:** MAE, RMSE, R² 

**Output:** `results/linear_*.png`


### Step 6: Evaluation - Complex Models
```bash
python src/evaluation_complex.py
```

**Purpose:** Evaluate ensemble model performance on test set (2024)

**Visualizations:** Predicted vs Actual scatter plots, residual analysis

**Output:** `results/complex_*.png`, `results/evaluation_results_timeaware.csv`

## Project Structure

```
Predicting-property-sale-prices/
│
├── .github/
│   └── workflows/
│       ├── pipeline.yml          # CI/CD workflow
│       └── pylint.yml            # Code quality check
│
├── data/
│   ├── raw/                      # DVF CSV files 
│   ├── processed/                # Processed dataset
│   │   ├── dvf_processed.parquet
│   │   └── dvf_featured.parquet
│   └── sample/                   # Sample data for CI/CD
│       └── dvf_featured_sample.parquet
│
├── models/                       # Trained model (.pkl)
│   ├── linear_regression.pkl
│   ├── ridge.pkl
│   ├── lasso.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── results/                      # Evaluation outputs
│   ├── evaluation_results_timeaware.csv
│   ├── evaluation_results.csv
│   └── *.png                     # Visualization plots
│
├── src/                          # Source code modules
│   ├── preprocessing.py          
│   ├── feature_engineering.py    
│   ├── linear_models.py          
│   ├── complex_models.py         
│   ├── evaluation_linear.py      
│   └── evaluation_complex.py     
│
├── notebooks/
│   └── Exploration_notebook.ipynb  # Initial exploration of data
│
├── .gitattributes                # Git LFS configuration
├── requirements.txt              # Python dependencies
├── PROPOSAL.md                   # Project proposal
├── report.pdf                    # Technical report
└── README.md                    
```

---

## CI/CD Pipeline

The project implements automated testing via **GitHub Actions**, executing the complete pipeline on every code push:

1. **Code quality validation:** Pylint analysis (8.04/10 score)
2. **Pipeline execution:** Feature engineering → training → evaluation on sample data
3. **Results archival:** Automated storage of metrics, visualizations, and trained models

**Environment-Aware Data Loading:**
- **CI environment:** Uses 100K training samples (GitHub Actions 7GB RAM limit)
- **Local environment:** Uses full dataset (10.9M training samples)

```python
import os

def load_data():
    """Load appropriate dataset based on execution environment."""
    is_ci = os.getenv('CI') == 'true'
    
    if is_ci:
        path = 'data/sample/dvf_featured_sample.parquet'
        print("CI environment: using 10k sample")
    else:
        path = 'data/processed/dvf_featured.parquet'
        print("Local environment: using full dataset")
    
    return pd.read_parquet(path)
```

## Code Quality

- **Pylint Score:** 8.04/10
- **Documentation:** Comprehensive docstrings following NumPy conventions
- **Type Hints:** Full function signature annotations
- **Modular Design:** Single-responsibility principle throughout
- **Version Control:** Git with LFS for efficient large file management

---

**Author:** Yvan Roh  
**Institution:** HEC Lausanne / University of Lausanne  
**Email:** yvan.roh@unil.ch  
**Student ID:** 22-216-170

---

**Last Updated:** November 2024
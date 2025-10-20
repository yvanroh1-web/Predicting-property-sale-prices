# Predicting Property Sale Prices in France Using Machine Learning: A Study Based on the DVF Dataset (2019–2024)


## Problem Statement and Motivation

The DVF (<i>Demandes de Valeurs Foncières</i>) dataset, published by the French Directorate-General of Public Finances, provides granular records of real estate transactions across France from 2019 to 2024 (excluding Alsace, Moselle, and Mayotte). Despite its richness, raw property data is noisy and high-dimensional, making it difficult for non-experts to extract actionable insights. This project aims to build a robust machine learning model that predicts sale prices using key features—such as property type, surface area, geographic location, and transaction date—while explicitly modeling temporal dynamics (e.g., market trends and seasonality). The resulting tool could support buyers, sellers, investors, and urban planners in making data-driven decisions.



\## Planned Approach and Technologies

The project will be implemented entirely in **Python** using the following stack:

- **Data handling**: `pandas`, `numpy`

- **Visualization**: `matplotlib`, `seaborn`

- **Engineering**: temporal feature extraction (month, quarter, rolling trends), one-hot encoding, outlier treatment

- **Modeling**: `scikit-learn` (Implement various regression models)

- **Evaluation**: MAE, R², time-aware cross-validatio



Key phases:


1. **Data preprocessing**: clean missing values, remove implausible outliers (e.g., price = 0 or surface > 10,000 m²).

2. **Feature engineering**: derive time-based features (e.g., year, month, days since 2019) and encode categorical variables.

3. **Exploratory Data Analysis (EDA)**: visualize price distributions, spatial patterns, and temporal trends.

4. **Model development**: train and compare baseline and time-aware regression models.

5. **Validation**: use time-series-aware splits (e.g., train on 2019–2022, validate on 2023–2024) to avoid look-ahead bias.



## Expected Challenges and Mitigation Strategies

- **Data volume and noise**: The DVF contains a lot of rows with inconsistent entries.  
  **Solution**: implement modular preprocessing with validation checks and sampling for prototyping.

- **Temporal leakage**: Standard cross-validation may leak future information.  
  **Solution**: use temporal train/validation splits and rolling window evaluation.

- **Geospatial granularity**: Commune-level data may lack precision.  
  **Solution**: focus on relative location (e.g., department or urban/rural flag) rather than exact coordinates.



## Success Criteria

The project will be considered successful if:

\- The final model achieves **MAE < €50,000** and **R² > 0.75** on out-of-sample 2023–2024 data.

\- All core components (preprocessing, modeling, evaluation) are **unit-tested** (>70% coverage).

\- The codebase is **modular, documented, and reproducible** via a `requirements.txt` and clear `README.md`.


---

> Dataset source: \[DVF on data.gouv.fr](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres/)


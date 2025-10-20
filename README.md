# Predicting Property Sale Prices in France

**Advanced Programming 2025 — HEC Lausanne**  
**Author:** Yvan Roh  
**Instructor:** Prof. Simon Scheidegger  
**Course:** Data Science & Advanced Programming (DSAP 2025)  
**Date:** December 2025  

---

# Abstract

<!--
This project explores the prediction of real estate sale prices in France using the DVF (Demandes de Valeurs Foncières) dataset published by the French Directorate-General of Public Finances.  
The main objective is to develop a data-driven predictive model that can estimate property prices based on structural, spatial and temporal features extracted from property transactions between 2019 and 2024.

The project involves a complete data science workflow, including data cleaning, feature engineering, exploratory analysis and model development.  
Preliminary findings suggest geographical and seasonal patterns in property values. This shows that a regression-based or tree-based model could effectively capture these dynamics.
-->
---

# Table of Contents

1. [Introduction](#1-introduction)  
2. [Literature Review](#2-literature-review)  
3. [Methodology](#3-methodology)  
4. [Results](#4-results)  
5. [Discussion](#5-discussion)  
6. [Conclusion](#6-conclusion)  
7. [References](#references)  
8. [Appendices](#appendices)

---

# 1. Introduction

## 1.1 Background and Motivation
The valuation of real estate properties is a critical aspect of financial stability and economic planning. <br>
Accurate price prediction models can support investors, policymakers, and real estate professionals by providing insights into market trends and anomalies. <br>
The **DVF dataset**, made available through the French government’s open data initiative, provides an excellent basis for such analysis.

## 1.2 Problem Statement
This project aims to **predict the sale price of real estate transactions** based on property characteristics and contextual information (e.g., type, size, location, date of sale).

## 1.3 Objectives
- Clean and preprocess the raw DVF dataset.  
- Conduct exploratory analysis to understand feature relationships.  
- Build a regression-based predictive model for property sale prices.  
- Evaluate performance using standard metrics such as MAE and R².

## 1.4 Structure of the Report
- **Section 2:** Literature Review  
- **Section 3:** Methodology  
- **Section 4:** Results and evaluation  
- **Section 5:** Discussion and limitations  
- **Section 6:** Conclusion and future work  

---

# 2. Literature Review

Previous studies on real estate valuation have explored various regression and machine learning approaches. <br>
Common methods include **hedonic pricing models**, **random forests**, and **gradient boosting regressors**, which have been shown to capture both linear and nonlinear relationships in housing markets.
<br>
However, few studies have focused specifically on **the DVF dataset**, which combines both geographic and temporal richness. <br>
This project thus extends existing work by integrating **spatial analysis and time-based feature engineering** into a unified predictive framework.

*Work in progress*

---

# 3. Methodology

## 3.1 Data Description
- **Source:** DVF dataset (2019–2024) — [data.gouv.fr](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres)  
- **Size:** Over 10 million property transactions  
- **Main Variables:**
  - `valeur_fonciere`: property sale price  
  - `surface_reelle_bati`: building surface area  
  - `type_local`: property type (house, apartment, dépendance, etc.)  
  - `date_mutation`: transaction date  
  - `longitude`, `latitude`: geographic coordinates  

The exploratory notebook present in `notebooks\exploration.ipynb` provides an overview of missing values, correlations, and spatial patterns.

## 3.2 Data Cleaning

---

## 3.3 Feature Engineering

---

## 3.4 Model Development & Evaluation
---

# 4. Results

---

# 5. Discussion

---

# 6. Conclusion

---

# References

1. **DVF Dataset.** (2024). *Demandes de Valeurs Foncières*. [data.gouv.fr](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres)  
2. HEC Lausanne (2025). *Advanced Programming Rulebook.* University of Lausanne.  

---

# Appendices



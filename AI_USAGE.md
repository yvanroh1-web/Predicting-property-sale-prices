# AI Usage

**Project:** Predicting Property Sale Prices in France using Machine Learning  
**Author:** Yvan Roh  
**Course:** Advanced Programming - HEC Lausanne  
**Date:** November 2024

---

## Use of Generative AI

AI was used at different stages of this project, adopting a critical and responsible approach to the responses obtained.

**Claude 3.5 Sonnet** ([https://claude.ai](https://claude.ai)) and **ChatGPT-4** ([https://chat.openai.com](https://chat.openai.com)) were used as coding and editorial assistants. It is important to emphasize that their use remained strictly limited to improving existing content, debugging assistance, and learning new concepts, without replacing personal reflection or original production.


## Main Assigned Tasks

AI was primarily used to:
- Assist with Python code syntax and debugging
- Help understand and implement machine learning concepts
- Verify code quality and suggest improvements following best practices
- Reformulate and correct certain initially written paragraphs in the project's report
- Improve the clarity and fluidity of documentation
- Assist with proper use of some libraries

## Specific AI Contributions

### 1. **Data Preprocessing Pipeline** (`src/preprocessing.py`)
- AI helped with initial structure for memory-efficient data type optimization
- I modified the `optimize_dtypes()` function to fit DVF dataset specifics

### 2. **Feature Engineering** (`src/feature_engineering.py`)
- Used AI to understand cyclical encoding (sin/cos) for temporal features
- AI suggested best practices for preventing data leakage in temporal splits
- I implemented and adapted the `add_department_features_safe()` function to compute geographic aggregations only on training data

### 3. **Model Training** (`src/linear_models.py`, `src/complex_models.py`)
- AI explained hyperparameter options for Random Forest and XGBoost
- Used AI to understand scikit-learn Pipeline architecture
- I designed the complete model comparison strategy and parameter choices

### 4. **CI/CD Pipeline** (`.github/workflows/pipeline.yml`)
- AI provided boilerplate GitHub Actions workflow structure
- I customized it for environment-aware data loading (CI vs local execution)
- Added specific steps for DVF dataset handling with Git LFS

### 5. **Test Suite and Code Quality**
- Used AI to understand pylint requirements and improve code.

### 6. **Report**
- AI helped reformulate complex technical explanations for clarity
- Used AI to verify academic writing style and structure
- All analytical work, methodology design, and conclusions are 100% original

## Examples of Typical Questions Used

**For Code Development:**
- "Explain the difference between bagging (Random Forest) and boosting (XGBoost) for regression tasks"
- "How to use scikit-learn's ColumnTransformer with mixed feature types?"
- "Best practices for saving and loading trained models with joblib"

**For Machine Learning Concepts:**
- "Why does XGBoost tend to overfit more easily than Random Forest on temporal data?"
- "How to properly implement temporal validation with chronological splits?"
- "Explain the difference between StandardScaler and no scaling for tree-based models"
- "What are the advantages of log transformation for highly skewed target variables?"

**For Technical Writing:**
- "Here's an extract from my methodology section: [text]. Can you help me make it clearer and more concise?"
- "How should I structure the discussion of a modular pipeline in an academic report?"
- "Is this explanation of feature engineering well-formulated?"

**For CI/CD and DevOps:**
- "How to configure GitHub Actions to detect execution environment?"
- "What's the best way to handle large data files with Git LFS in automated workflows?"
- "How to upload artifacts (results, plots, models) in GitHub Actions?"

## Critical Approach

The responses provided by AI tools were analyzed, validated, and integrated in a thoughtful manner. No code or text was used as-is without personal understanding and modification.

**Key Principles:**
- All AI suggestions were reviewed, understood, and validated by the author
- Code was written or substantially modified with full comprehension of every line
- All analytical work, model selection decisions, and conclusions are 100% original
- All feature engineering strategies were designed by me based on domain knowledge
- AI served as an assistant and learning tool, not a content generator
- Every algorithm and approach can be explained in detail

## Learning Moments

Through interaction with AI tools, I gained deeper understanding of:

1. **Temporal Data Leakage Prevention**: Understanding why computing features on the full dataset before splitting causes leakage, and how to properly implement leak-safe feature engineering.

2. **Model-Specific Feature Selection**: Learning why different model families (linear vs tree-based) benefit from different preprocessing strategies (standardization, encoding methods).

3. **Pipeline Architecture**: Understanding how to build modular, reproducible ML pipelines using ColumnTransformer and Pipeline objects.

4. **CI/CD Best Practices**: Understanding how to design environment-aware code that adapts to different execution contexts (local development vs automated testing).

**Note:**

The tools of AI have been invaluable learning companion throughout this project and have helped me understand key concepts: gradient boosting, temporal validation, and organization of code. However, core intellectual work in this project-from formulation of the problem to experimental design through final conclusions-represents my learning journey and technical growth in data science and machine learning.
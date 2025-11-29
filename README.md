# Telco Customer Churn Prediction

End-to-end analysis and modeling to predict churn in a telco subscription business. The notebook walks through data understanding, cleaning, class-imbalance handling, feature engineering, model comparison, and hyperparameter tuning to deliver a high-recall churn classifier.

## Project Structure
- `telco_customer_churn_prediction.ipynb` — full workflow (EDA → preprocessing → modeling).
- `README.md` — project overview and usage.

## Dataset
- Source: IBM/Kaggle “Telco Customer Churn” dataset (convert the Excel file to CSV and save as `customer_churn.csv` in the repo root).
- Target: `churn_label` (converted to binary `is_churned`).
- Columns removed: identifier  `customerid`   

## Environment Setup
- Python 3.10+ (tested on 3.12).
- Install deps:
  ```bash
  pip install pandas numpy matplotlib seaborn mlxtend plotly scikit-learn imbalanced-learn xgboost
  ```
- Launch Jupyter:
  ```bash
  jupyter notebook telco_customer_churn_prediction.ipynb
  ```

## How to Run
1. Place `customer_churn.csv` in the project root.
2. Open the notebook and run cells in order.
3. Grid searches for hyperparameter tuning may take a few minutes.

## Methodology
- EDA: churn rates by customer segments; higher churn observed for higher monthly charges and specific service/contract patterns.
- Cleaning: standardized column names, dropped constant/duplicate rows, converted target to binary.
- Splits: stratified 80/20 train/test; 5-fold cross-validation for model comparison.
- Preprocessing: median imputation for numeric, standard scaling, one-hot encoding for categorical features, 3-sigma outlier handling on training data.
- Class imbalance: SMOTE-ENN pipeline inside modeling to oversample minority class and clean noise.
- Feature selection: mutual information explored; retained all features to avoid performance loss.

## Models & Results
- Models evaluated: Logistic Regression, KNN, Gradient Boosting, Random Forest, XGBoost (all with cross-val).
- Tuning: grid search per model on the preprocessed, balanced pipeline.
- Winning model: **Random Forest** after hyperparameter tuning.
  - Test accuracy: 0.7544
  - Test recall (churn): 0.8952
  - Test F1: 0.6588
  - Test ROC-AUC: 0.8747
  - Notes: high recall prioritized to catch churners; precision trade-off acceptable for retention workflows.

## Next Steps
- Package the pipeline (preprocess + model) into a deployable artifact (e.g., `joblib` or FastAPI microservice).

# Credit Scoring Model

This project implements a machine learning model to predict credit risk based on customer financial behavior and demographic data. The model evaluates the likelihood of a customer experiencing serious delinquency (90+ days past due) within two years.

## Features

- **Data Preprocessing**: Handles missing values, outlier detection, and feature scaling
- **Feature Engineering**: Creates new predictive features like:
  - Log_Income
  - Total_Late_Payments
  - Debt_to_Income_Bin
  - Has_Dependents
- **Class Imbalance Handling**: Uses SMOTENC for synthetic minority oversampling
- **Model Comparison**: Evaluates four classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- **Threshold Optimization**: Finds optimal classification thresholds using F1 and Youden's index
- **Explainability**: Includes SHAP values for model interpretability

## Results Summary

| Model                | ROC-AUC | F1-Score | Precision | Recall |
|----------------------|---------|----------|-----------|--------|
| Logistic Regression  | 0.812   | 0.585    | 0.562     | 0.626  |
| Decision Tree        | 0.816   | 0.594    | 0.575     | 0.613  |
| Random Forest        | 0.824   | 0.598    | 0.564     | 0.635  |
| XGBoost              | 0.824   | 0.598    | 0.592     | 0.605  |

![ROC Curves](roc_curves.png)

## How to Use

1. **Install Dependencies**
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib seaborn
```

2. **Run the Notebook**
```bash
jupyter notebook Credit_Scoring_Model.ipynb
```

3. **Dataset Requirements**
- Place `credit_scoring_sample.csv` in the working directory
- Ensure the dataset contains these required columns:
  - `SeriousDlqin2yrs`
  - `age`
  - `MonthlyIncome`
  - `DebtRatio`
  - `NumberOfTime30-59DaysPastDueNotWorse`
  - `NumberOfTimes90DaysLate`
  - `NumberOfTime60-89DaysPastDueNotWorse`
  - `NumberOfDependents`

## Key Features Created

| Feature Name               | Description |
|----------------------------|-------------|
| `Log_Income`               | Natural log of monthly income |
| `Total_Late_Payments`      | Sum of all late payment counts |
| `Debt_to_Income_Bin`       | Quartile bins for debt-to-income ratio |
| `Has_Dependents`           | Binary flag for having dependents |

## Model Evaluation Metrics
- **ROC-AUC**: Area under Receiver Operating Characteristic curve
- **PR-AUC**: Area under Precision-Recall curve
- **F1-Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Accuracy adjusted for class imbalance
- **Confusion Matrix**: Visualizes TP, FP, TN, FN

## Dependencies
- Python 3.7+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- shap
- matplotlib
- seaborn

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.
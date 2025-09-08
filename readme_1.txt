# Breast Cancer Diagnosis Classification Model

## Overview
This Jupyter notebook implements a machine learning pipeline for classifying breast cancer tumors as benign (B) or malignant (M) using the Wisconsin Breast Cancer Dataset. The approach includes data preprocessing, feature selection, model training with a stacking ensemble, evaluation, interpretability analysis using SHAP, and error analysis.

Key highlights:
- **Dataset**: Breast cancer features from `breast_cancer_data.csv` (569 samples, 32 features including diagnosis).
- **Feature Selection**: Recursive Feature Elimination (RFE) with Random Forest to select top 10 features.
- **Models**: Stacking ensemble combining Logistic Regression, SVM (RBF kernel), Random Forest, XGBoost, and MLP, with Logistic Regression as the meta-learner.
- **Evaluation Metrics**: Accuracy, ROC-AUC, Classification Report, Cross-Validation.
- **Interpretability**: SHAP values for feature importance using Random Forest on selected features.
- **Performance**: Achieves ~97.37% test accuracy and ~99.80% ROC-AUC on the test set.

The selected top 10 features by RFE are: `['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'concave points_worst']`.

Top features by Random Forest importance:
| Feature              | Importance |
|----------------------|------------|
| perimeter_worst      | 0.203265  |
| radius_worst         | 0.167324  |
| concave points_mean  | 0.152399  |
| area_worst           | 0.150038  |
| concave points_worst | 0.148496  |
| concavity_mean       | 0.044933  |
| texture_worst        | 0.040733  |
| radius_mean          | 0.035493  |
| area_mean            | 0.033928  |
| perimeter_mean       | 0.023392  |

## Requirements
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap`

Install via pip:
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

## Dataset
- Download the Wisconsin Breast Cancer Dataset (UCI repository) as `breast_cancer_data.csv`.
- Features: 30 numerical features (e.g., radius_mean, texture_mean) + diagnosis label ('M' for malignant, 'B' for benign).
- No missing values; 'id' column is dropped.

## Usage
1. Place `breast_cancer_data.csv` in the same directory as the notebook.
2. Open `Untitled29.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells sequentially.
   - Cells 1-3: Imports, data loading, and train-test split (80/20 stratified).
   - Cell 4: RFE for feature selection (top 10 features).
   - Cell 5: Scaling with StandardScaler.
   - Cells 6-8: Define base models and stacking ensemble pipeline.
   - Cell 9: Train pipeline and evaluate (prints accuracy, CV scores, ROC-AUC, classification report).
   - Cells 10-13: SHAP analysis on Random Forest (summary plot displayed).
   - Cell 14: SHAP summary plot.
   - Cell 15: Error analysis (misclassified samples).
   - Cell 16: Feature importance table.

### Expected Output
- Test Accuracy: ~0.9737
- CV Accuracy (5-fold): ~0.9582 (±0.0189)
- ROC-AUC: ~0.9980
- Misclassified samples: Typically 3 indices (e.g., [73, 86, 205]).

## Pipeline Details
- **Preprocessing**: Map diagnosis to binary (1=M, 0=B), drop 'id', train-test split with stratification.
- **Feature Selection**: RFE with RandomForestClassifier (n_features=10).
- **Scaling**: StandardScaler on selected features.
- **Ensemble**:
  - Base: LR (liblinear), SVM (RBF, prob=True), RF (200 est., depth=10), XGB (100 est., depth=3), MLP (100 hidden, 200 iter).
  - Meta: Logistic Regression.
  - CV: 5-fold StratifiedKFold.
- **Evaluation**: Accuracy, ROC-AUC, precision-recall, confusion matrix via cross_val_score and sklearn metrics.
- **SHAP**: TreeExplainer on RF model for malignant class interpretability (beeswarm plot).

## Error Analysis
- Misclassified samples are identified by comparing predictions to true labels.
- Example: First misclassified sample features are printed (e.g., radius_mean=13.80, etc.).

## Potential Improvements
- Hyperparameter tuning (e.g., GridSearchCV) for base models.
- Additional models (e.g., LightGBM).
- Handle class imbalance if present (though stratified split is used).
- Deploy as a web app (e.g., via Streamlit or Flask).

## License
This project is for educational purposes. Dataset from UCI ML Repository (public domain).

For questions or contributions, contact the author.
# Loan Approval Prediction with Explainable AI

This project predicts whether a loan applicant is likely to be approved or rejected, then explains the model's decision using SHAP.

## Dataset

- Source: Loan Prediction Problem Dataset
- Rows: 614 training applicants
- Target: `Loan_Status`
- Positive class: approved loan application

## Project Workflow

1. `notebooks/01_Data_Understanding.ipynb`
   - Loads train and test data
   - Checks structure, missing values, categorical columns, and target balance

2. `notebooks/02_Data_Cleaning.ipynb`
   - Fills missing values
   - Encodes categorical features
   - Drops `Loan_ID`
   - Saves `data/cleaned_train.csv`

3. `notebooks/03_Model_Training.ipynb`
   - Trains Logistic Regression, Random Forest, and XGBoost
   - Compares Accuracy, Precision, Recall, F1, and ROC-AUC
   - Saves the selected model to `models/logistic_regression_model.pkl`

4. `notebooks/04_SHAP_Explainability.ipynb`
   - Loads the trained model
   - Builds SHAP explanations
   - Shows global and individual prediction explanations

5. `notebooks/05_Predict_New_Applicant.ipynb`
   - Loads the saved model and expected feature columns
   - Predicts approval for a new applicant
   - Explains the individual prediction with SHAP

## Project Structure

```text
Loan_Default_Prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── cleaned_train.csv
├── models/
│   └── logistic_regression_model.pkl
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_SHAP_Explainability.ipynb
│   └── 05_Predict_New_Applicant.ipynb
├── README.md
└── requirements.txt
```

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run the notebooks in order from the `notebooks/` directory.

## Key Result

Logistic Regression performed best in the saved workflow. SHAP showed that `Credit_History` is the strongest driver of the model's approval predictions.

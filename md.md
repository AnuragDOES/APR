# Heart Diresease Classification with Logistic Regression

## Overview

This project applies logistic regression to the Cleveland Heart Disease dataset to predict heart disease presence based on patient health records. All tasks are performed in Python using pandas and scikit-learn

## Dataset

- **Source:** Kaggle, Cleveland Heart Disease (`cherngs/heart-disease-cleveland-uci`)
- **File Used:** `heart_cleveland_upload.csv`
- **Target Variable:** `condition` (presence of heart disease)
- Dataset was downloaded using `kagglehub`.


## Methodology

- Data loaded and features/target selected.
- Train-test split (80/20) with stratification.
- Feature scaling via StandardScaler.
- Logistic Regression model trained (`max_iter=5000`).
- Model evaluations: accuracy, confusion matrix, classification report (precision, recall, F1-score).


## Results

Key metrics from model evaluation:

- **Accuracy:** 91.66%
- **Confusion Matrix:**

```
[[32  0]
 [ 5 23]]
```

- **Classification Report:**

```
            precision    recall  f1-score   support

      0       0.86      1.00      0.93        32
      1       1.00      0.82      0.90        28

  accuracy                            0.92        60
  macro avg       0.93      0.91      0.91        60
  weighted avg    0.93      0.92      0.92        60
```


## How to Run

1. Ensure Python >=3.7, pandas, scikit-learn, and kagglehub are installed.
2. Set up your Kaggle API credentials for downloading datasets.
3. Run the `main.py` script (provided in this repo).

## Files Included

- `main.py` — main code with all steps
- `README.md` — explanation and summary
- `report.pdf` — screenshots of code and outputs

***






# Project 10: Fraud Detection Analysis
### Finding the 0.17%: Detecting Fraudulent Transactions in Highly Imbalanced Data

---

## Business Brief

Credit card fraud costs the global economy over 30 billion dollars annually. The challenge is not building a model that detects fraud. Anyone can do that by flagging every transaction as fraud and achieving 100% recall. The real challenge is catching fraud without drowning legitimate customers in false alarms.

This is a class imbalance problem. Fraudulent transactions represent 0.17% of all transactions. A naive model predicting everything as legitimate achieves 99.83% accuracy and catches zero fraud.

---

## Dataset

| Property | Detail |
|----------|--------|
| **Name** | Credit Card Fraud Detection |
| **Direct Link** | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| **Records** | 284,807 transactions over 2 days |
| **Fraud rate** | 0.172% (492 fraud cases) |
| **Features** | 28 PCA-transformed features + Amount + Time |

---

## What Makes This Different

- Explicitly demonstrates why accuracy is the wrong metric before building anything
- Applies SMOTE to training data only (common mistake is contaminating the test set)
- Evaluates with PR-AUC, the correct metric for severe class imbalance
- Compares Logistic Regression vs Random Forest on equal ground
- Builds a cost analysis framework and finds the cost-optimal threshold
- Shows that 0.5 is almost never the right threshold for fraud detection

---

## Project Structure

```
fraud-detection-analysis/
├── project_10_fraud_detection.ipynb
└── README.md
```

---

## Kaggle Setup

1. Search **"creditcardfraud"** by *mlg-ulb* on Kaggle and attach it
2. Upload `project_10_fraud_detection.ipynb`
3. Run the path finder cell first
4. Run all cells

Note: This notebook requires `imbalanced-learn` for SMOTE. Kaggle has it pre-installed. If running locally, install with `pip install imbalanced-learn`.

---

## Notebook Walkthrough

### Section 1: Setup
Libraries including imbalanced-learn for SMOTE. Colour system: red = fraud, green = legitimate.

### Section 2: Load Data
Loads the dataset. Prints exact fraud and legitimate counts with percentages. Shows the ratio: for every 1 fraud case there are approximately 577 legitimate transactions.

### Section 3: Exploratory Analysis
Three-panel chart: transaction amount distribution by class, class imbalance bar chart with percentages, amount statistics table (count, mean, std, min, max for both classes).

### Section 4: Why Accuracy is Wrong
Demonstrates the naive baseline: a model predicting everything as legitimate achieves 99.83% accuracy but catches zero fraud. Explains why PR-AUC and recall are the correct metrics.

### Section 5: Prepare Data and Handle Imbalance
Scales Amount and Time with StandardScaler. Stratified train/test split. Applies SMOTE to training set only with explicit note about why test contamination is a common mistake. Falls back to class_weight if imbalanced-learn is unavailable.

### Section 6: Train and Compare Models
Trains Logistic Regression and Random Forest on resampled data. Evaluates both on the untouched test set using ROC-AUC, PR-AUC, F1, Precision, and Recall. Identifies the best model by PR-AUC.

### Section 7: Model Evaluation Dashboard
Three-panel chart: ROC curves for both models, Precision-Recall curves for both models with baseline, confusion matrix for the best model.

### Section 8: Cost-Optimal Threshold
Tests every threshold from 0.01 to 0.99 (200 steps). Calculates total cost at each threshold using: false negatives cost full fraud amount, false positives cost 5% of transaction amount (investigation). Finds the threshold that minimises total cost. Compares savings vs the default 0.5 threshold. Two charts: cost curve and precision/recall/F1 vs threshold.

### Section 9: Feature Importance
Top 15 features from the best model. Random Forest uses feature_importances_, Logistic Regression uses absolute coefficient values.

### Section 10: Executive Summary Dashboard
Dark-theme KPI cards: total transactions, fraud rate, best model PR-AUC, recall at default threshold, cost-optimal threshold, recall at optimal threshold.

### Section 11: Findings and Conclusions
Five findings with evidence. Three key lessons: stop reporting accuracy, threshold is a business decision, SMOTE contamination warning.

---

## Key Concepts Covered

| Concept | Where Used |
|---------|-----------|
| Class imbalance problem | Sections 2 and 4 |
| SMOTE oversampling | Section 5 |
| PR-AUC vs ROC-AUC | Sections 6 and 7 |
| Threshold tuning | Section 8 |
| Cost-benefit analysis | Section 8 |
| Feature importance | Section 9 |

---

## Tools Used

| Tool | Purpose |
|------|---------|
| Python (Pandas, NumPy) | Data manipulation |
| Scikit-learn | Models, metrics, preprocessing |
| Imbalanced-learn | SMOTE oversampling |
| Plotly | Interactive dashboard |
| Matplotlib + Seaborn | Evaluation charts, confusion matrix |

---

*Built by Jessica Dan-Odhomo - [LinkedIn](https://www.linkedin.com/in/jessica-dan-odhomo) - [GitHub](https://github.com/Teekaayyy)*

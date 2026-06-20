# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will **churn**
(cancel their service) based on demographic, account, and billing data,
comparing Decision Trees, Random Forest, and Boosting methods.

---

## Dataset

- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size:** 7,043 customers, 21 columns
- **Target:** `Churn` → `Yes` / `No`
- **Features:** demographics, services subscribed, contract type, billing info

### Class Balance
No (stayed):  5,174  (73.5%)

Yes (churned): 1,869  (26.5%)
Moderately imbalanced — minority class (churned customers) is harder to predict.

---

## Data Cleaning

- `TotalCharges` was stored as text (`object`) with **11 blank entries** (`' '`)
- Converted to numeric with `pd.to_numeric(errors='coerce')`
- Dropped the 11 resulting null rows → final dataset: **7,032 rows**
- No other missing values in the dataset

---

## Exploratory Data Analysis

- **Count plot** of churn distribution
- **Violin & box plots** of `TotalCharges` by churn status
- **Box plot** of `TotalCharges` by `Contract` type, split by churn
- **Correlation bar chart** — one-hot encoded features vs `Churn_Yes`
- **Histogram** of `tenure` (customer lifetime in months)
- **Facet grid (`displot`)** — tenure distribution by `Contract` × `Churn`
- **Scatterplot** of `MonthlyCharges` vs `TotalCharges`, colored by churn
- **Churn rate by tenure** — calculated and plotted month-by-month:
  - Month 1: **62%** churn rate
  - Month 72: **1.7%** churn rate
  - Clear trend: **longer tenure → dramatically lower churn**

### Tenure Cohorts (engineered feature)

```python
def cohort(tenure):
    if tenure < 13:  return '0-12 Months'
    elif tenure < 25: return '12-24 Months'
    elif tenure < 49: return '24-48 Months'
    else: return 'Over 48 Months'
```

- Visualized churn by cohort with count plots and a faceted plot by `Contract`

---

## Preprocessing

- Dropped `Churn` and `customerID` from features → `X`
- **One-hot encoded** categorical features (`pd.get_dummies(drop_first=True)`)
- Split: **90% training / 10% test** (`test_size=0.1, random_state=101`)

---

##  Models Compared

### 1. Decision Tree (max_depth=6)

```python
DecisionTreeClassifier(max_depth=6)
```

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| No | 0.87 | 0.89 | 0.88 |
| Yes | 0.55 | 0.49 | 0.52 |
| **Accuracy** | | | **0.81** |

**Top Feature Importances:**

| Feature | Importance |
|---------|-----------|
| `tenure` | **0.424** |
| `InternetService_Fiber optic` | 0.314 |
| `TotalCharges` | 0.062 |
| `MonthlyCharges` | 0.046 |
| `PaymentMethod_Electronic check` | 0.034 |
| `Contract_Two year` | 0.027 |
| `OnlineSecurity_No internet service` | 0.026 |

> `tenure` and `Fiber optic internet` dominate — together they explain ~74% of the tree's decisions.

---

### 2. Random Forest

```python
RandomForestClassifier()
```

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| No | 0.86 | 0.88 | 0.87 |
| Yes | 0.50 | 0.46 | 0.48 |
| **Accuracy** | | | **0.79** |

> Slightly underperformed the single Decision Tree here — default RF wasn't tuned.

---

### 3. AdaBoost (n_estimators=100)  Best Model

```python
AdaBoostClassifier(n_estimators=100)
```

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| No | 0.88 | 0.92 | 0.90 |
| Yes | 0.62 | 0.50 | 0.56 |
| **Accuracy** | | | **0.83** |

> Best overall accuracy and best churn-class F1-score among all models.

---

### 4. Gradient Boosting

```python
GradientBoostingClassifier()
```

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| No | 0.87 | 0.90 | 0.89 |
| Yes | 0.57 | 0.50 | 0.53 |
| **Accuracy** | | | **0.82** |

> Close second to AdaBoost.

---

## Model Comparison Summary

| Model | Accuracy | Churn (Yes) F1 |
|-------|----------|-----------------|
| Decision Tree (depth=6) | 0.81 | 0.52 |
| Random Forest (default) | 0.79 | 0.48 |
| Gradient Boosting | 0.82 | 0.53 |
| **AdaBoost (n=100)** | **0.83** | **0.56** |

---

## Libraries Used

- `numpy`, `pandas` — data handling
- `matplotlib`, `seaborn` — visualization (countplot, violin, box, histplot, displot, scatterplot, catplot)
- `scikit-learn`:
  - `DecisionTreeClassifier`, `RandomForestClassifier`
  - `AdaBoostClassifier`, `GradientBoostingClassifier`
  - `train_test_split`
  - `classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay`


---

## Key Takeaways

| Insight | Detail |
|---------|--------|
| Best model | **AdaBoost** (83% accuracy, 0.56 F1 for churn) |
| Strongest churn predictor | **Tenure** — new customers churn far more than long-term ones |
| Second strongest predictor | **Fiber optic internet service** |
| Churn rate trend | Drops from ~62% (month 1) to ~2% (month 72) |
| Class imbalance impact | All models struggle more with the minority "Yes" (churn) class |

> **Lesson 1:** Customer tenure is by far the strongest churn signal — retention
> efforts should focus heavily on the first 12 months of the customer lifecycle.

> **Lesson 2:** Boosting methods (AdaBoost, Gradient Boosting) outperformed
> bagging (Random Forest) on this imbalanced dataset, likely due to their
> focus on hard-to-classify minority examples.

> **Lesson 3:** With imbalanced classes, accuracy alone is insufficient —
> always evaluate F1-score and recall for the minority class.

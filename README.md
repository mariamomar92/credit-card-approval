# Credit Card Approval Prediction

Predicting whether a credit-card client ends up **past due** on their payments, using their application details and their monthly credit history.

The project merges two datasets on a shared client `ID`, cleans and explores the data, runs a set of statistical tests, and then trains and compares three classifiers (Logistic Regression, Random Forest, and XGBoost).

---

## Dataset

The data comes from the **[Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)** dataset on Kaggle. It is made up of two files:

| File | What's in it |
|------|--------------|
| `application_record.csv` | One row per client: gender, income, education, family status, housing, employment length, age (in days), and several flags. |
| `credit_record.csv` | Monthly credit records per client (`MONTHS_BALANCE`) with a payment `STATUS`. |

The two are joined on the `ID` column. `STATUS` is collapsed into a simple two-class target: **past due** vs **no past due**.

### A note on the data

The exact origin of this dataset isn't clearly documented, it may be anonymized real banking records or simulated data, the public sources don't say definitively. Either way it contains **no personal identifiers** (no names, addresses, phone numbers, or account numbers, just a surrogate `ID`), so no individual can be identified from it.

As a precaution, and to respect the dataset's license, the raw CSV files are **not included in this repository**. To run the notebook:

1. Download the dataset from the Kaggle link above.
2. Create a `data/` folder in the project root.
3. Put `application_record.csv` and `credit_record.csv` inside it.

Please check the dataset's license on its Kaggle page before redistributing it anywhere.

---

## Project structure

```
.
├── credit_card_prediction.ipynb   # the full analysis
├── README.md
├── requirements.txt
└── data/                          # you add the CSVs here (downloaded from Kaggle)
    ├── application_record.csv
    └── credit_record.csv
```

---

## What the notebook does

The notebook is organized into clear sections:

1. **A first look at the data** — shape, column types, and missing values.
2. **Cleaning** — filling missing occupation values, dropping duplicates, building the two-class target, and fixing the day/month columns that are stored as negatives.
3. **Handling outliers** — capping extreme values with the IQR rule, with before/after boxplots.
4. **Hypothesis testing** — formal tests instead of eyeballing the plots:
   - t-tests (income by gender; income and months-balance for past-due vs not)
   - one-way ANOVA with Tukey HSD (income by education, housing, and occupation)
   - ANOVA on past-due rate by occupation
   - a two-way ANOVA (education × occupation)
   - Mann–Whitney U test (income by car ownership)
5. **Exploratory data analysis** — distributions and group comparisons (gender split, car ownership, occupation, income by education, ownership status, past-due trend over time, age by occupation).
6. **Encoding & feature prep** — converting categoricals to numeric, a correlation heatmap, and dropping non-informative columns (IDs and the constant phone flag).
7. **Balancing & modelling** — oversampling the minority class, scaling, then training the three models.

---

## Models

Three classifiers are trained and compared on the same metrics — **accuracy, precision, recall, F1, and a confusion matrix**:

- Logistic Regression
- Random Forest
- XGBoost

Because the target is imbalanced (most clients are not past due), the training data is oversampled with `RandomOverSampler` and the features are standardized before fitting.

The Random Forest and XGBoost hyperparameters were tuned; the values shown in the notebook are the final selected ones rather than library defaults.

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|:--------:|:---------:|:------:|:--:|
| Logistic Regression | 0.56 | 0.56 | 0.55 | 0.55 |
| Random Forest | 0.98 | 0.95 | 1.00 | 0.98 |
| XGBoost | 0.95 | 0.92 | 0.99 | 0.95 |

Because the classes are imbalanced, recall and F1 on the past-due class are more meaningful here than raw accuracy. The tree-based models score far higher than logistic regression on this split.

One caveat worth being honest about: the class balancing is applied before the train/test split, so duplicated rows can appear in both sets. That inflates the tree-model scores (a recall of 1.00 is a sign of this), so they should be read as optimistic. Splitting the data first and balancing only the training set would give a more trustworthy estimate.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
imbalanced-learn
xgboost
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## How to run

1. Clone the repo and add the data as described above.
2. Install the requirements.
3. Open `credit_card_prediction.ipynb` in Jupyter (or VS Code) and run the cells top to bottom.

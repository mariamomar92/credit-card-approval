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
7. **Balancing & modelling** — splitting the data, scaling on the training set only, handling the class imbalance with class weights, then training the three models and comparing them (including ROC / precision-recall curves and AUC).
---
 
## Models
 
Three classifiers are trained and compared on the same metrics — **accuracy, precision, recall, F1, and a confusion matrix**:
 
- Logistic Regression
- Random Forest
- XGBoost
Because the target is imbalanced (most clients are not past due), the imbalance is handled with `class_weight='balanced'` (and `scale_pos_weight` for XGBoost) rather than oversampling. To avoid data leakage, the train/test split is done first and the scaler is fit on the training set only, then applied to the test set.
 
The Random Forest and XGBoost hyperparameters were tuned; the values shown in the notebook are the final selected ones rather than library defaults.
 
---
 
## Results
 
Evaluated on a held-out test set that keeps the real (imbalanced) class ratio:
 
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|:------:|
| Logistic Regression | 0.56 | 0.02 | 0.55 | 0.04 | 0.574 | 0.020 |
| Random Forest | 0.96 | 0.19 | 0.51 | 0.28 | 0.895 | 0.265 |
| XGBoost | 0.92 | 0.13 | 0.69 | 0.21 | 0.886 | 0.271 |
 
Only about 1–2% of clients are actually past due, so **accuracy is misleading here** — a model that simply predicted "never past due" would already score high. Precision at the default 0.5 threshold is low because, on such a rare class, the models raise a lot of false positives.
 
The more informative number is **ROC-AUC**: the tree models reach ≈ 0.89, meaning they rank a risky client above a safe one about 89% of the time. So while they aren't useful as a blunt yes/no classifier at the default threshold, they do what a credit-risk model actually needs — order clients by risk. Logistic regression (ROC-AUC 0.57) is close to random and barely captures the signal. PR-AUC tells the same story: the tree models sit well above the ~0.02 baseline while logistic regression does not.
 
Random Forest and XGBoost perform similarly; XGBoost catches more past-due clients (higher recall) while Random Forest raises slightly fewer false alarms.
 
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


# XGBoost
# Binary Classification: Within vs. Not Within
# Nested Repeated Stratified 5-Fold Cross-Validation

import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load dataset

df = pd.read_csv("FinalClaudeUnder.csv")

# Remove rows with missing target
df = df.dropna(subset=['Pricing Outcome (calc)']).copy()

# Create binary target
df['Status'] = (
    df['Pricing Outcome (calc)']
    .str.strip()
    .str.title()
    .apply(lambda x: 'Within' if x == 'Within' else 'Not Within')
)

# Check target distribution
print("\nTarget Distribution:")
print(df['Status'].value_counts())

# Encode binary target:
# Not Within = 0
# Within = 1

y = df['Status'].map({
    'Not Within': 0,
    'Within': 1
})

# Create six predictors

# Convert dates
df['IPO Date'] = pd.to_datetime(
    df['IPO Date'], errors='coerce'
)

df['SEC S-1 filing date'] = pd.to_datetime(
    df['SEC S-1 filing date'], errors='coerce'
)

# Calculate days between S-1 filing and IPO
df['S-1 to IPO (days)'] = (
    df['IPO Date'] - df['SEC S-1 filing date']
).dt.days

# Calculate filed range midpoint
df['Range Midpoint'] = (
    df['Filed Range Low ($)'] +
    df['Filed Range High ($)']
) / 2

# Calculate pre-pricing deal size
df['Pre-pricing Deal Size ($M)'] = (
    df['Range Midpoint'] *
    df['Shares Offered (M)']
)

# Select the same six predictors
features = [
    'Total Funding ($M)',
    'Num of Funding Rounds (PRE-IPO)',
    'S-1 to IPO (days)',
    'Pre-pricing Deal Size ($M)',
    'underwriter_count',
    'Range Midpoint'
]

X = df[features]

print("\nPredictors:")
print(features)

print("\nNumber of observations:", len(df))

print("\nClass Encoding:")
print("Not Within = 0")
print("Within = 1")

# Create XGBoost pipeline

model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('xgb', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=1
    ))
])

# Define hyperparameter tuning

param_grid = {
    'xgb__n_estimators': [50, 100, 200],
    'xgb__max_depth': [1, 2, 3],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__subsample': [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0]
}

# Inner cross-validation: 3 folds
inner_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# Define outer cross-validation

# Outer cross-validation: 5 folds, 3 repeats
outer_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

# Run nested cross-validation

accuracy_scores = []
balanced_accuracy_scores = []
f1_scores = []

baseline_accuracy_scores = []
baseline_balanced_scores = []

all_actual = []
all_predicted = []

best_parameters = []

for fold, (train_index, test_index) in enumerate(
    outer_cv.split(X, y), start=1
):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # Tune hyperparameters using training data only
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='balanced_accuracy',
        refit=True,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Best model selected by inner CV
    best_model = grid_search.best_estimator_

    # Predict on untouched outer test fold
    y_pred = best_model.predict(X_test)

    # Evaluate model
    accuracy_scores.append(
        accuracy_score(y_test, y_pred)
    )

    balanced_accuracy_scores.append(
        balanced_accuracy_score(y_test, y_pred)
    )

    # F1-score specifically for Within (class 1)
    f1_scores.append(
        f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )
    )

    # Majority-class baseline using training data
    majority_class = y_train.mode()[0]

    baseline_pred = np.full(
        len(y_test), majority_class
    )

    baseline_accuracy_scores.append(
        accuracy_score(y_test, baseline_pred)
    )

    baseline_balanced_scores.append(
        balanced_accuracy_score(y_test, baseline_pred)
    )

    # Save predictions and best parameters
    all_actual.extend(y_test.tolist())
    all_predicted.extend(y_pred.tolist())

    best_parameters.append(grid_search.best_params_)

    print(
        f"Completed Fold {fold}/15 | "
        f"Best parameters: {grid_search.best_params_}"
    )

# Display overall performance

print("\n========================================")
print("XGBOOST RESULTS")
print("Binary Classification: Within vs. Not Within")
print("Nested Repeated Stratified 5-Fold CV")
print("3 Repeats | 15 Outer Evaluations")
print("========================================")

print("\nModel Performance:")

print(
    "Average Accuracy:",
    round(np.mean(accuracy_scores), 4)
)

print(
    "Accuracy Std:",
    round(np.std(accuracy_scores, ddof=1), 4)
)

print(
    "Average Balanced Accuracy:",
    round(np.mean(balanced_accuracy_scores), 4)
)

print(
    "Balanced Accuracy Std:",
    round(np.std(balanced_accuracy_scores, ddof=1), 4)
)

print(
    "Average F1 (Within):",
    round(np.mean(f1_scores), 4)
)

print(
    "F1 Std:",
    round(np.std(f1_scores, ddof=1), 4)
)

# Majority-class baseline

print("\nMajority-Class Baseline:")

print(
    "Average Accuracy:",
    round(np.mean(baseline_accuracy_scores), 4)
)

print(
    "Average Balanced Accuracy:",
    round(np.mean(baseline_balanced_scores), 4)
)

# Confusion matrix

labels = [0, 1]
class_names = ['Not Within', 'Within']

cm = confusion_matrix(
    all_actual,
    all_predicted,
    labels=labels
)

print("\nConfusion Matrix:")
print("Rows = Actual | Columns = Predicted")

print(pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
))

print("\nConfusion Matrix Interpretation:")
print(f"True Not Within:  {cm[0, 0]}")
print(f"False Within:     {cm[0, 1]}")
print(f"False Not Within: {cm[1, 0]}")
print(f"True Within:      {cm[1, 1]}")

# Classification report

print("\nClassification Report:")

print(
    classification_report(
        all_actual,
        all_predicted,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )
)

# Hyperparameter selection

print("\nBest Hyperparameters Across Outer Folds:")

for i, params in enumerate(best_parameters, start=1):
    print(f"Fold {i}: {params}")

print("\nMost Frequently Selected Parameter Combinations:")

print(
    pd.Series([
        str(params) for params in best_parameters
    ]).value_counts()
)

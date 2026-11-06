import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)

data = pd.read_excel("CS 441.xlsx", sheet_name="FINAL_CLAUDE")

price_cols = ["IPO Share Price ($)", "Filed Range Low ($)", "Filed Range High ($)"]
for col in price_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.dropna(subset=price_cols).copy()

# 0 = within range, 1 = outside range
data["outside"] = ((data["IPO Share Price ($)"] < data["Filed Range Low ($)"]) | (data["IPO Share Price ($)"] > data["Filed Range High ($)"])).astype(int)

# new columns
data["midpoint"] = (data["Filed Range Low ($)"] + data["Filed Range High ($)"]) / 2
data["deal_size"] = data["midpoint"] * pd.to_numeric(data["Shares Offered (M)"], errors="coerce")

# count lead underwriters
def count_underwriters(value):
    if pd.isna(value): return None
    text = str(value)
    extra = re.findall(r"\(([^)]*?)-\s*lead mgr\)", text, flags=re.IGNORECASE)
    outside = re.sub(r"\([^)]*\)", "", text)
    names = [name.strip() for name in outside.split(",") if name.strip()]
    names += [name.strip() for name in extra if name.strip()]
    return len(names) if names else None

data["underwriter_count"] = data["Lead Underwriters"].apply(count_underwriters)

features = ["Total Funding ($M)", "Num of Funding Rounds (PRE-IPO)","days_s1_to_ipo", "deal_size", "underwriter_count", "midpoint"]

for col in features: data[col] = pd.to_numeric(data[col], errors="coerce")

X = data[features]
y = data["outside"]

print("Number of IPOs:", len(data))
print("0 = within, 1 = outside")
print(y.value_counts().sort_index())
print("\nMissing values:")
print(X.isna().sum())

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, C=0.1))

baseline = DummyClassifier(strategy="most_frequent")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# predictions
all_preds = []
all_probs = []
base_preds = []

for repeat in range(3):
    all_preds.append([None] * len(data))
    all_probs.append([None] * len(data))
    base_preds.append([None] * len(data))

for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
    repeat = fold // 5
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    base = baseline.predict(X_test)

    for i, row in enumerate(test_index):
        all_preds[repeat][row] = preds[i]
        all_probs[repeat][row] = probs[i]
        base_preds[repeat][row] = base[i]

accuracy = []
balanced = []
f1 = []
auc = []
base_accuracy = []
base_balanced = []

for repeat in range(3):
    preds = all_preds[repeat]
    probs = all_probs[repeat]
    base = base_preds[repeat]
    accuracy.append(accuracy_score(y, preds))
    balanced.append(balanced_accuracy_score(y, preds))
    f1.append(f1_score(y, preds, zero_division=0))
    auc.append(roc_auc_score(y, probs))
    base_accuracy.append(accuracy_score(y, base))
    base_balanced.append(balanced_accuracy_score(y, base))

print("\nLogistic Regression Results")
print("Accuracy:", round(sum(accuracy) / 3, 3))
print("Balanced accuracy:", round(sum(balanced) / 3, 3))
print("F1 (outside):", round(sum(f1) / 3, 3))
print("ROC-AUC:", round(sum(auc) / 3, 3))
print("Baseline accuracy:", round(sum(base_accuracy) / 3, 3))
print("Baseline balanced accuracy:", round(sum(base_balanced) / 3, 3))

print("\nResults by repetition:")
for repeat in range(3):
    print("Repeat", repeat + 1,
          "- Accuracy:", round(accuracy[repeat], 3),
          "Balanced accuracy:", round(balanced[repeat], 3))

# confusion matrix
cm = confusion_matrix(y, all_preds[0], labels=[0, 1])
ConfusionMatrixDisplay(cm, display_labels=["Within", "Outside"]).plot(cmap="Blues")
plt.title("Logistic Regression - 5-Fold CV (Repeat 1)")
plt.tight_layout()
plt.savefig("logistic_confusion_matrix.png")
plt.close()

results = pd.DataFrame({
    "Company": data["Companies"],
    "Actual": y,
    "Predicted": all_preds[0],
    "Probability Outside": all_probs[0]})
results.to_csv("logistic_predictions.csv", index=False)

# coefficients
model.fit(X, y)
print("\nCoefficients:")
for name, value in zip(features, model.named_steps["logisticregression"].coef_[0]): print(name, round(value, 3))

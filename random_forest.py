"""
Random Forest - IPO pricing signals (CS 441, Team 2)
Predicts whether a biotech IPO prices Within (0) or Outside (1) its filed price range.

Evaluation: stratified 5-fold CV repeated 3 times (15 outer folds); inner stratified
3-fold GridSearchCV (balanced accuracy) on training folds only. Median imputation is
inside the pipeline so it is fit on training data only.

Usage:
    python random_forest.py path/to/CS_441_Final_dataset_.csv            # main results
    python random_forest.py path/to/CS_441_Final_dataset_.csv --ablation # + top-k predictor check (slow)
Outputs: printed results, rf_per_company_predictions.csv
"""
import sys
import collections
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "CS_441_Final_dataset_.csv"
RUN_ABLATION = "--ablation" in sys.argv

# ---------------------------------------------------------------- data
d = pd.read_csv(CSV).iloc[:33].copy()          # 33 IPOs; drops blank row + note row at the bottom
d["mid"] = (d["Filed Range Low ($)"] + d["Filed Range High ($)"]) / 2
d["deal"] = d["mid"] * d["Shares Offered (M)"]

X = pd.DataFrame({
    "Total Funding ($M)": d["Total Funding ($M)"],
    "Pre-IPO Funding Rounds": d["Num of Funding Rounds (PRE-IPO)"],
    "S-1 to IPO Days": d["days_s1_to_ipo"],
    "Pre-pricing Deal Size ($M)": d["deal"],
    "Lead Underwriter Count": d["underwriter_count"],
    "Range Midpoint ($)": d["mid"],
})
y = (d["price_vs_range"] != "within").astype(int).values   # 0 = Within, 1 = Outside
print("X:", X.shape, "| Within/Outside:", np.bincount(y))

GRID = {
    "rf__max_depth": [1, 2, 3],
    "rf__n_estimators": [100],
    "rf__min_samples_leaf": [1, 2],
}


def make_search():
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(random_state=0, class_weight="balanced", n_jobs=1)),
    ])
    return GridSearchCV(pipe, GRID, cv=StratifiedKFold(3, shuffle=True, random_state=1),
                        scoring="balanced_accuracy")


def evaluate(cols):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    rows, cm_all, cm_rep0 = [], np.zeros((2, 2), int), np.zeros((2, 2), int)
    chosen, imps = collections.Counter(), []
    prob_sum, correct, seen = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        gs = make_search().fit(X.iloc[tr][cols], y[tr])
        pred = gs.predict(X.iloc[te][cols])
        prob = gs.predict_proba(X.iloc[te][cols])[:, 1]
        rows.append([
            accuracy_score(y[te], pred),
            balanced_accuracy_score(y[te], pred),
            f1_score(y[te], pred, average="macro"),
            f1_score(y[te], pred, pos_label=1),
            roc_auc_score(y[te], prob),
        ])
        cm = confusion_matrix(y[te], pred, labels=[0, 1])
        cm_all += cm
        if i < 5:                                   # first repetition = first 5 folds
            cm_rep0 += cm
        chosen[tuple(sorted(gs.best_params_.items()))] += 1
        imps.append(gs.best_estimator_.named_steps["rf"].feature_importances_)
        prob_sum[te] += prob
        correct[te] += (pred == y[te])
        seen[te] += 1
        print(f"  fold {i + 1}/15", flush=True)
    return (np.array(rows), cm_all, cm_rep0, chosen, np.mean(imps, axis=0),
            prob_sum / seen, correct.astype(int))


cols = list(X.columns)
r, cm_all, cm_rep0, chosen, imp, p_out, n_correct = evaluate(cols)

names = ["Accuracy", "Balanced Accuracy", "Macro F1", "F1 (Outside)", "ROC-AUC"]
print("\n=== Headline scores (mean +/- sd over 15 outer folds) ===")
for n, m, s in zip(names, r.mean(0), r.std(0)):
    print(f"{n:18s} {m:.4f} +/- {s:.4f}")
print("Baseline: accuracy 0.5762, balanced accuracy 0.5000, ROC-AUC 0.500")

print("\n=== Confusion matrix, all folds (rows actual Within/Outside; cols predicted Within/Outside) ===")
print(cm_all)
print(f"recall Within {cm_all[0,0]/cm_all[0].sum():.3f} | recall Outside {cm_all[1,1]/cm_all[1].sum():.3f}")
print("Repeat 0 only:\n", cm_rep0)

print("\n=== Selected hyperparameters per fold ===")
for k, v in chosen.most_common():
    print(v, dict(k))

print("\n=== Mean feature importance ===")
imp_s = pd.Series(imp, index=cols).sort_values(ascending=False)
print(imp_s.round(3))

out = pd.DataFrame({
    "company": d["Companies"].str.replace(r"\s*\(.*\)", "", regex=True).values,
    "outcome": d["price_vs_range"].values,
    "actual_outside": y,
    "mean_oof_prob_outside": p_out.round(3),
    "correct_predictions_of_3": n_correct,
})
out.to_csv("rf_per_company_predictions.csv", index=False)
print("\nSaved rf_per_company_predictions.csv")

if RUN_ABLATION:
    print("\n=== Did more predictors help? (top-k by importance) ===")
    print("Note: ranking uses importances from the same data, so this is slightly optimistic.")
    for k in [2, 3, 4]:
        rr = evaluate(list(imp_s.index[:k]))[0]
        print(k, list(imp_s.index[:k]), "acc/bal/macroF1/F1out/AUC:", rr.mean(0).round(4))
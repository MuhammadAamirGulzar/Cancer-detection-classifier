import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from datetime import datetime

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

FEATURE_ROOT = "__"
LABELS_PATH = "__"

MODELS = {
   
}

CLASSIFIERS = {
    "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "SVM": SVC(kernel="rbf", class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

N_SPLITS = 5
OUTPUT_DIR = f"full_classifier_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ────────────────────────────────────────────────
# OPTIONAL: AGGREGATION STRATEGIES
# ────────────────────────────────────────────────

# def aggregate_features(features, slide_ids, strategy="mean"):
#     df = pd.DataFrame(features)
#     df["slide_id"] = slide_ids

#     if strategy == "mean":
#         return df.groupby("slide_id").mean().values
#     elif strategy == "max":
#         return df.groupby("slide_id").max().values
#     else:
#         raise ValueError("Unknown aggregation strategy")


# ────────────────────────────────────────────────
# EVALUATION FUNCTION
# ────────────────────────────────────────────────

def cross_validate_classifier(clf, X, y, groups=None):

    if groups is None:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        splits = cv.split(X, y)
    else:
        cv = GroupKFold(n_splits=N_SPLITS)
        splits = cv.split(X, y, groups)

    acc_scores = []
    bacc_scores = []
    f1_scores = []

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        bacc_scores.append(balanced_accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average="macro"))

    return (
        np.mean(acc_scores), np.std(acc_scores),
        np.mean(bacc_scores), np.std(bacc_scores),
        np.mean(f1_scores), np.std(f1_scores)
    )


# ────────────────────────────────────────────────
# MAIN PIPELINE
# ────────────────────────────────────────────────

def main():

    df_labels = pd.read_csv(LABELS_PATH)

    # Required columns:
    # label
    # slide_id (if slide-level evaluation)
    labels = df_labels["label"].values
    slide_ids = df_labels.get("slide_id", None)

    results = []

    for model_name, feature_file in MODELS.items():

        print("\n" + "="*70)
        print(f"Evaluating Features from {model_name}")
        print("="*70)

        features = np.load(os.path.join(FEATURE_ROOT, feature_file))

        for clf_name, clf in CLASSIFIERS.items():

            print(f"\nClassifier: {clf_name}")

            if slide_ids is not None:
                groups = slide_ids.values
            else:
                groups = None

            acc_m, acc_s, bacc_m, bacc_s, f1_m, f1_s = \
                cross_validate_classifier(clf, features, labels, groups)

            results.append({
                "Feature_Model": model_name,
                "Classifier": clf_name,
                "Accuracy_mean": acc_m,
                "Accuracy_std": acc_s,
                "BalancedAcc_mean": bacc_m,
                "BalancedAcc_std": bacc_s,
                "F1_macro_mean": f1_m,
                "F1_macro_std": f1_s
            })

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(OUTPUT_DIR, "final_comparison.csv"), index=False)

    print("\nFINAL SUMMARY")
    print(summary.round(4))
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
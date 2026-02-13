!pip -q install openpyxl seaborn scikit-learn

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, classification_report
)


# ---------------------------
# CONFIG
# ---------------------------
FILE_PATH = "lung_disease_v1.xlsx"   # change if needed
RANDOM_STATE = 42
TEST_SIZE = 0.20

# These are the numerical columns you want to summarize (edit if your file differs)
NUM_SUMMARY_COLS = [
    "age", "bmi", "pack_years", "fev1_fvc",
    "ct_nodule_size_mm", "crp_mg_L", "ct_emphysema_pct"
]

# ---------------------------
# Helper: OneHotEncoder compatibility
# ---------------------------
def make_onehot_encoder():
    """
    scikit-learn >= 1.2 uses sparse_output
    older versions use sparse
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ---------------------------
# MAIN
# ---------------------------
def main():
    # ---------- Load ----------
    df = pd.read_excel(FILE_PATH)

    print("=== DATA LOADED ===")
    print("Shape:", df.shape)

    if "disease_type" not in df.columns:
        raise ValueError("Column 'disease_type' not found. Please check your dataset columns.")

    print("\n=== TARGET DISTRIBUTION (disease_type) ===")
    print((df["disease_type"].value_counts(normalize=True).round(4) * 100).astype(str) + "%")

    # ---------- Descriptive stats ----------
    existing_num_cols = [c for c in NUM_SUMMARY_COLS if c in df.columns]
    if existing_num_cols:
        print("\n=== DESCRIPTIVE STATISTICS (selected numerical columns) ===")
        print(df[existing_num_cols].describe().round(2))
    else:
        print("\n[Warning] None of the specified NUM_SUMMARY_COLS exist in this file.")

    # ---------- Plots ----------
    # If you want plots saved instead of shown, replace plt.show() with plt.savefig("name.png")
    sns.set_style("whitegrid")

    # Plot 1: Target distribution
    plt.figure(figsize=(10, 5))
    order = df["disease_type"].value_counts().index
    sns.countplot(data=df, x="disease_type", order=order)
    plt.title("Distribution of Disease Types")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 2: Boxplot of fev1_fvc by disease_type (only if fev1_fvc exists)
    if "fev1_fvc" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="disease_type", y="fev1_fvc", order=order)
        plt.title("FEV1/FVC Ratio by Disease Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("\n[Info] Column 'fev1_fvc' not found → skipping FEV1/FVC boxplot.")

    # Plot 3: Correlation heatmap (numeric-only)
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] >= 2:
        plt.figure(figsize=(12, 8))
        corr = num_df.corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.2)
        plt.title("Correlation Heatmap – Numerical Features")
        plt.tight_layout()
        plt.show()
    else:
        print("\n[Info] Not enough numeric columns for correlation heatmap.")

    # ---------- Preprocess ----------
    print("\n=== PREPROCESSING ===")

    # Handle missing family_history if present
    if "family_history" in df.columns:
        df["family_history"] = df["family_history"].fillna("None")

    # Features & target
    drop_cols = [c for c in ["patient_id", "disease_type", "severity"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["disease_type"]

    # Identify numeric & categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", make_onehot_encoder())
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Transform
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    print("Train shape after preprocessing:", X_train_prep.shape)
    print("Test  shape after preprocessing:", X_test_prep.shape)

    # ---------- Models ----------
    print("\n=== TRAINING MODELS ===")

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs"
        ),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "KNN": KNeighborsClassifier()
    }

    for name, clf in models.items():
        print(f"Training {name} ...")
        clf.fit(X_train_prep, y_train)

    # ---------- Evaluate ----------
    print("\n=== EVALUATION (Test Set) ===")
    results = []

    for name, clf in models.items():
        y_pred = clf.predict(X_test_prep)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (macro)": round(prec, 4),
            "Recall (macro)": round(rec, 4),
            "F1 (macro)": round(f1, 4)
        })

    results_df = pd.DataFrame(results).sort_values("F1 (macro)", ascending=False)
    print(results_df.to_string(index=False))

    # Best model by macro F1
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    print(f"\nBest model by Macro-F1: {best_model_name}")

    # Optional: full classification report for best model
    y_best = best_model.predict(X_test_prep)
    print("\n=== Classification Report (Best Model) ===")
    print(classification_report(y_test, y_best, zero_division=0))

    # ---------- Confusion Matrix for RandomForest (as you asked) ----------
    if "RandomForest" in models:
        print("\n=== CONFUSION MATRIX: RandomForest (normalized) ===")
        fig, ax = plt.subplots(figsize=(9, 7))
        ConfusionMatrixDisplay.from_estimator(
            models["RandomForest"],
            X_test_prep,
            y_test,
            normalize="true",
            values_format=".2%",
            cmap="Blues",
            ax=ax
        )
        plt.title("Normalized Confusion Matrix – Random Forest")
        plt.tight_layout()
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
# ==============================================================
# MODULE 2 — DATA PREPROCESSING SCRIPT
# Project: A Hybrid Machine Learning Model for Efficient XML Parsing
# ==============================================================

# 🔹 Step 1 — Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# ==============================================================
# 🔹 Step 2 — Load the dataset
# ==============================================================

filename = "backend/xml_profiling_data(Module1_results).csv"

if not os.path.exists(filename):
    raise FileNotFoundError(f"⚠️ File '{filename}' not found in current directory!")

df = pd.read_csv(filename, encoding="utf-8", on_bad_lines='skip')
print("✅ Dataset loaded successfully!\n")

# ==============================================================
# 🔹 Step 3 — Basic Info
# ==============================================================

print("Dataset Info:")
print(df.info())
print("\nSample Data:\n", df.head(), "\n")

# ==============================================================
# 🔹 Step 4 — Missing / Duplicate Checks
# ==============================================================

print("Missing values in each column:\n", df.isnull().sum(), "\n")
print("Duplicate rows found:", df.duplicated().sum(), "\n")

df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# ==============================================================
# 🔹 Step 5 — Encode Target Column
# ==============================================================

if "Efficient_Algo" not in df.columns:
    raise KeyError("⚠️ Column 'Efficient_Algo' not found in dataset!")

le = LabelEncoder()
df["Efficient_Algo_Label"] = le.fit_transform(df["Efficient_Algo"])
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nLabel Encoding Map:", label_map, "\n")

# ==============================================================
# 🔹 Step 6 — Define Features (X) and Target (y)
# ==============================================================

feature_cols = ['File Size (MB)','No of Tags','XML Depth','No of Attributes','No of Elements','CPU Cores','Memory Usage (MB)']
missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    raise KeyError(f"⚠️ Missing feature columns: {missing_cols}")

X = df[feature_cols]
y = df["Efficient_Algo_Label"]

# ==============================================================
# 🔹 Step 7 — Train-Test Split (Preserving ALL Classes)
# ==============================================================

label_counts = y.value_counts()
print("\nClass Distribution:\n", label_counts, "\n")

# Check if stratification is possible
min_class_count = label_counts.min()
if min_class_count >= 2:
    # Stratified split possible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("✅ Stratified split applied successfully.\n")
else:
    # Use random split to preserve all samples including rare classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print("⚠️ Random split applied (rare classes with <2 samples detected).\n")

print(f"✅ Data Split Completed:\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}\n")
print(f"Training set class distribution:\n{y_train.value_counts()}\n")
print(f"Testing set class distribution:\n{y_test.value_counts()}\n")

# ==============================================================
# 🔹 Step 8 — Feature Standardization
# ==============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling done.\n")

# ==============================================================
# 🔹 Step 9 — Convert Scaled Data Back to DataFrames (Optional)
# ==============================================================

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)

# ==============================================================
# 🔹 Step 10 — Save Processed Data
# ==============================================================

np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

processed_df = pd.concat([df[feature_cols], df['Efficient_Algo_Label']], axis=1)
processed_df.to_csv("module2_preprocessed.csv", index=False, encoding="utf-8")

print("💾 Files Saved:")
print(" - X_train_scaled.npy")
print(" - X_test_scaled.npy")
print(" - y_train.npy")
print(" - y_test.npy")
print(" - module2_preprocessed.csv\n")

print("✅ Preprocessing completed successfully! Ready for ANN + SVM model training.")
print(f"✅ All {len(label_map)} classes preserved in the dataset.")
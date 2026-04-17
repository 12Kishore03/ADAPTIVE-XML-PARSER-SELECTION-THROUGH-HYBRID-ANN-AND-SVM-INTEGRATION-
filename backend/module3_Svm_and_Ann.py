# ==============================================================
# IMPROVED MODULE 3 — HYBRID MODEL WITH ENHANCED FEATURES
# Addresses Low Accuracy by Adding Feature Engineering
# ==============================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ==============================================================
# Step 1 — Load Preprocessed Data
# ==============================================================

data_file = "module2_preprocessed.csv"

if not os.path.exists(data_file):
    raise FileNotFoundError("❌ Preprocessed file not found!")

data = pd.read_csv(data_file)
print("✅ Preprocessed data loaded successfully.")
print(f"Total samples: {len(data)}")

# ==============================================================
# Step 2 — CRITICAL: FEATURE ENGINEERING
# ==============================================================
# 🔴 THIS IS THE KEY TO IMPROVING ACCURACY! 🔴

print("\n🔧 [NEW] Creating Enhanced Features...")

# Original features assumed to exist:
# File Size (MB), No of Tags, XML Depth, No of Attributes, 
# No of Elements, CPU Cores, Memory Usage (MB)

# RATIO FEATURES (help model understand relationships)
data['tags_per_mb'] = data['No of Tags'] / (data['File Size (MB)'] + 0.01)
data['elements_per_mb'] = data['No of Elements'] / (data['File Size (MB)'] + 0.01)
data['attrs_per_element'] = data['No of Attributes'] / (data['No of Elements'] + 1)
data['memory_efficiency'] = data['Memory Usage (MB)'] / (data['File Size (MB)'] + 0.01)
data['memory_per_core'] = data['Memory Usage (MB)'] / data['CPU Cores']

# COMPLEXITY FEATURES (capture XML structure complexity)
data['structural_complexity'] = (data['XML Depth'] * data['No of Attributes']) / (data['No of Elements'] + 1)
data['data_density'] = (data['No of Tags'] + data['No of Elements']) / (data['File Size (MB)'] + 0.01)
data['depth_complexity'] = data['XML Depth'] / (data['File Size (MB)'] + 1)

# INTERACTION FEATURES (core count affects performance differently based on file size)
data['cores_x_filesize'] = data['CPU Cores'] * data['File Size (MB)']
data['cores_x_memory'] = data['CPU Cores'] * data['Memory Usage (MB)']

# POLYNOMIAL FEATURES (non-linear relationships)
data['filesize_squared'] = data['File Size (MB)'] ** 2
data['cores_squared'] = data['CPU Cores'] ** 2

# LOGARITHMIC FEATURES (handle wide value ranges)
data['log_filesize'] = np.log1p(data['File Size (MB)'])
data['log_tags'] = np.log1p(data['No of Tags'])
data['log_elements'] = np.log1p(data['No of Elements'])

# CATEGORICAL FEATURES BASED ON YOUR RULES
def categorize_file_size(size):
    if size <= 4: return 0      # 0-4 MB
    elif size <= 25: return 1   # 4-25 MB
    elif size <= 60: return 2   # 25-60 MB
    elif size <= 100: return 3  # 60-100 MB
    else: return 4              # 100-155 MB

def categorize_cores(cores):
    if cores <= 2: return 0     # Low (DOM/JDOM territory)
    elif cores <= 8: return 1   # Medium
    else: return 2              # High (SAX territory)

data['size_category'] = data['File Size (MB)'].apply(categorize_file_size)
data['core_category'] = data['CPU Cores'].apply(categorize_cores)

# COMBINED DECISION FEATURE (mimics your algorithm selection rules)
data['size_core_combo'] = (data['size_category'] * 10) + data['core_category']

print(f"✅ Created {len([col for col in data.columns if col not in ['Efficient_Algo_Label']]) - 7} new features")

# ==============================================================
# Step 3 — Prepare Training Data with Scaling
# ==============================================================

if 'Efficient_Algo_Label' not in data.columns:
    raise KeyError("❌ 'Efficient_Algo_Label' column not found!")

X = data.drop('Efficient_Algo_Label', axis=1)
y = data['Efficient_Algo_Label']

# Check class distribution
print("\n📊 Class Distribution:")
class_counts = y.value_counts().sort_index()
for label, count in class_counts.items():
    algo_names = {0: 'DOM', 1: 'JDOM', 2: 'PXTG', 3: 'SAX', 4: 'StAX'}
    print(f"   {algo_names.get(label, f'Label {label}')}: {count} ({count/len(y)*100:.1f}%)")

# 🔴 IMPORTANT: Scale features for better ANN/SVM performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
print(f"✅ Total features: {X_train.shape[1]}")

# ==============================================================
# Step 4 — IMPROVED ANN Architecture
# ==============================================================

num_features = X_train.shape[1]
num_classes = len(np.unique(y_train))

y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# 🔴 IMPROVED: Deeper network with batch normalization
ann = Sequential([
    Dense(128, activation='relu', input_dim=num_features),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    
    Dense(num_classes, activation='softmax')
])

# Custom optimizer with learning rate
optimizer = Adam(learning_rate=0.001)
ann.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🧠 Training Improved ANN...")
print(f"   Architecture: {num_features} → 128 → 64 → 32 → 16 → {num_classes}")

# 🔴 IMPROVED: Better callbacks
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Train with more epochs and callbacks
history = ann.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=200,  # Increased from 100
    batch_size=32,  # Increased from 16
    verbose=1,
    callbacks=[early_stop, reduce_lr]
)

# Evaluate ANN
ann_loss, ann_acc = ann.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✅ ANN Accuracy: {ann_acc * 100:.2f}%")

# Get predictions to see per-class performance
y_pred_ann = np.argmax(ann.predict(X_test, verbose=0), axis=1)
print("\n📊 ANN Classification Report:")
algo_names = ['DOM', 'JDOM', 'PXTG', 'SAX', 'StAX']
print(classification_report(y_test, y_pred_ann, target_names=algo_names))

# Extract features for SVM
get_features = Sequential(ann.layers[:-1])
X_train_ann_features = get_features.predict(X_train, verbose=0)
X_test_ann_features = get_features.predict(X_test, verbose=0)

# ==============================================================
# Step 5 — IMPROVED SVM with Hyperparameter Tuning
# ==============================================================

print("\n⚙️ Training Improved SVM on ANN features...")

# 🔴 IMPROVED: Better SVM parameters
svm = SVC(
    kernel='rbf',
    C=10.0,  # Increased from 1.0
    gamma='scale',
    probability=True,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

svm.fit(X_train_ann_features, y_train)

# Evaluate SVM
y_pred_svm = svm.predict(X_test_ann_features)
svm_acc = accuracy_score(y_test, y_pred_svm)

print(f"\n✅ SVM Accuracy (Hybrid Final): {svm_acc * 100:.2f}%")

print("\n📊 Final Hybrid Model Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=algo_names))

print("\n🧩 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_svm)
print("\nPredicted →  ", "  ".join([f"{name:>6s}" for name in algo_names]))
for i, row in enumerate(cm):
    print(f"{algo_names[i]:6s}      {row}")

# ==============================================================
# Step 6 — Feature Importance Analysis
# ==============================================================

print("\n" + "="*70)
print("📈 FEATURE IMPORTANCE INSIGHTS")
print("="*70)

# Calculate feature correlation with target
feature_importance = {}
for col in X.columns:
    correlation = abs(data[col].corr(data['Efficient_Algo_Label']))
    feature_importance[col] = correlation

sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print("\n🔝 Top 15 Most Important Features:")
for i, (feature, importance) in enumerate(sorted_features[:15], 1):
    print(f"   {i:2d}. {feature:30s} : {importance:.4f}")

# ==============================================================
# Step 7 — Save Models and Scaler
# ==============================================================

# Save scaler (CRITICAL for inference!)
scaler_path = "feature_scaler.joblib"
joblib.dump(scaler, scaler_path)
print(f"\n💾 Feature Scaler saved to: {scaler_path}")

# Save ANN Feature Extractor
ann_features_path = "hybrid_ann_features_extractor.keras"
get_features.save(ann_features_path)
print(f"💾 ANN Feature Extractor saved to: {ann_features_path}")

# Save SVM Classifier
svm_classifier_path = "hybrid_svm_classifier.joblib"
joblib.dump(svm, svm_classifier_path)
print(f"💾 SVM Classifier saved to: {svm_classifier_path}")

# ==============================================================
# Step 8 — Improvement Recommendations
# ==============================================================

print("\n" + "="*70)
print("💡 RECOMMENDATIONS TO FURTHER IMPROVE ACCURACY")
print("="*70)
"""print(""
1. ✅ DATASET QUALITY (MOST IMPORTANT!):
   • Generate MORE data (aim for 5000+ samples)
   • Ensure BALANCED classes (each algorithm ~20% of data)
   • Verify labels match the algorithm selection rules exactly
   • Remove outliers and inconsistent data

2. ✅ FEATURE ENGINEERING (Already Added!):
   • Ratio features (tags_per_mb, memory_efficiency)
   • Complexity metrics (structural_complexity)
   • Interaction terms (cores_x_filesize)
   • Categorical binning (size_category, core_category)

3. ✅ MODEL ARCHITECTURE:
   • Try ensemble methods (Random Forest, XGBoost)
   • Experiment with different ANN architectures
   • Use cross-validation for robust evaluation

4. ✅ HYPERPARAMETER TUNING:
   • Use GridSearchCV for SVM (C, gamma parameters)
   • Tune ANN learning rate, dropout, layers
   • Try different batch sizes and epochs

5. ✅ CLASS IMBALANCE HANDLING:
   • Use SMOTE (Synthetic Minority Oversampling)
   • Apply class_weight='balanced' in models
   • Consider focal loss for ANN

Expected Accuracy After Improvements: 75-90%+
"")"""

#print("\n✨ Improved Module 3 Complete!")
print(f"📊 Final Hybrid Model Accuracy: {svm_acc * 100:.2f}%")
print("="*70)
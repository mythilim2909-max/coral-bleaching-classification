import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---------------------------------------------------------
# STEP 1: Load the real environmental dataset, drop rows with missing data
# (the 1,568 land-masked Maui images we found earlier)
# ---------------------------------------------------------
df = pd.read_csv("./data/processed/coral_labels_with_environment.csv")
df = df.dropna(subset=['max_dhw_trailing_12wk', 'mean_sst_trailing_12wk', 'mean_sst_anomaly_trailing_12wk'])
print(f"Usable rows after dropping missing environmental data: {len(df)}")

# ---------------------------------------------------------
# STEP 2: Use the SAME train/test split as your image models
# WHY THIS MATTERS: for a fair comparison later (image-only vs env-only vs fusion),
# all three models must be tested on the exact same held-out images.
# ---------------------------------------------------------
print("\nSplit sizes in this environmental dataset:")
print(df['split'].value_counts())

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

feature_cols = ['max_dhw_trailing_12wk', 'mean_sst_trailing_12wk', 'mean_sst_anomaly_trailing_12wk']
X_train = train_df[feature_cols]
y_train = (train_df['label'] == 'bleached').astype(int)
X_test = test_df[feature_cols]
y_test = (test_df['label'] == 'bleached').astype(int)

print(f"\nTraining on {len(X_train)} images, testing on {len(X_test)} images.")

# ---------------------------------------------------------
# STEP 3: Baseline sanity check -- what if we just always guessed "healthy"?
# WHY: this tells us the minimum bar any real model must beat to be useful
# ---------------------------------------------------------
majority_class_accuracy = max(y_test.mean(), 1 - y_test.mean())
print(f"\n'Always guess majority class' baseline accuracy: {majority_class_accuracy*100:.2f}%")
print("Our real model must beat this by a meaningful margin to be useful.")

# ---------------------------------------------------------
# STEP 4: Train the environment-only model
# Using Random Forest -- this matches standard practice in existing coral
# bleaching prediction papers (e.g. Jagadeesh et al. 2025 used RF/XGBoost
# for exactly this kind of environmental-only prediction task)
# ---------------------------------------------------------
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 5: Evaluate on the real, unseen test set
# ---------------------------------------------------------
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Environment-only model test accuracy: {test_accuracy*100:.2f}%")
print("\nDetailed report:")
print(classification_report(y_test, y_pred, target_names=['healthy', 'bleached']))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# STEP 6: Which environmental feature mattered most?
# ---------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importance (which environmental signal mattered most):")
print(importances)

# ---------------------------------------------------------
# STEP 7: Save the model and the test set for later comparison
# ---------------------------------------------------------
joblib.dump(model, "./data/processed/environment_only_model.pkl")
test_df.to_csv("./data/processed/environment_test_set.csv", index=False)

print("\n✅ Saved model to ./data/processed/environment_only_model.pkl")
print("This accuracy number is baseline #2 (of 3) for your final comparison table.")
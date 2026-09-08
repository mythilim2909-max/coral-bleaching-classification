import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("./data/processed/coral_labels_rich_environment.csv")

feature_cols = [
    'max_dhw_trailing_12wk', 'mean_dhw_trailing_12wk', 'days_dhw_gt_4', 'days_dhw_gt_8',
    'mean_sst_trailing_12wk', 'max_sst_trailing_12wk', 'std_sst_trailing_12wk',
    'mean_sst_anomaly_trailing_12wk', 'max_sst_anomaly_trailing_12wk',
    'mean_hotspot_trailing_12wk', 'max_hotspot_trailing_12wk', 'max_baa_trailing_12wk',
    'dhw_trend'
]
df = df.dropna(subset=feature_cols)
print(f"Usable rows with all 13 rich features: {len(df)}")

unique_combos = df[feature_cols].drop_duplicates()
print(f"Unique feature combinations now: {len(unique_combos)} (compare to 60 before)\n")

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']
X_train, X_test = train_df[feature_cols], test_df[feature_cols]
y_train = (train_df['label'] == 'bleached').astype(int)
y_test = (test_df['label'] == 'bleached').astype(int)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Richer Random Forest -- Train: {train_acc*100:.2f}%  Test: {test_acc*100:.2f}%")
print(f"(Compare to old 3-feature version: Train 66.62%, Test 66.73%)")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, model.predict(X_test)))

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importance ranking:")
print(importances)
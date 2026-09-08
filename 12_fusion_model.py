import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---------------------------------------------------------
# STEP 1: Load both halves, verify they're actually in sync
# ---------------------------------------------------------
manifest_df = pd.read_csv("./data/processed/fusion_ready_manifest.csv")
image_features = np.load("./data/processed/image_features.npy")

print(f"Manifest rows: {len(manifest_df)}")
print(f"Image feature matrix shape: {image_features.shape}")
assert len(manifest_df) == image_features.shape[0], "Row count mismatch! STOP - paste this to me."
print("✅ Row counts match — safe to merge by position.\n")

# ---------------------------------------------------------
# STEP 2: The actual merge -- glue image clues + environmental clues together
# ---------------------------------------------------------
env_feature_cols = ['max_dhw_trailing_12wk', 'mean_sst_trailing_12wk', 'mean_sst_anomaly_trailing_12wk']
env_features = manifest_df[env_feature_cols].values

combined_features = np.hstack([image_features, env_features])
print(f"Combined feature matrix shape: {combined_features.shape}")
print(f"({image_features.shape[1]} image numbers + {env_features.shape[1]} environmental numbers = {combined_features.shape[1]} total per image)\n")

labels = (manifest_df['label'] == 'bleached').astype(int).values

# ---------------------------------------------------------
# STEP 3: Same train/test split as everything else, for a fair comparison
# ---------------------------------------------------------
train_mask = (manifest_df['split'] == 'train').values
test_mask = (manifest_df['split'] == 'test').values

X_train_raw, y_train = combined_features[train_mask], labels[train_mask]
X_test_raw, y_test = combined_features[test_mask], labels[test_mask]
print(f"Training on {len(X_train_raw)} images, testing on {len(X_test_raw)} images.\n")

# ---------------------------------------------------------
# STEP 4: Scale features so image numbers and temperature numbers are comparable
# ---------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ---------------------------------------------------------
# STEP 5: Train the fusion model -- the "Lead Doctor"
# ---------------------------------------------------------
fusion_model = MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=500, random_state=42, early_stopping=True)
fusion_model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 6: Evaluate on real, unseen test data
# ---------------------------------------------------------
y_pred = fusion_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"✅ FUSION MODEL test accuracy: {test_accuracy*100:.2f}%\n")
print("Detailed report:")
print(classification_report(y_test, y_pred, target_names=['healthy', 'bleached']))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# STEP 7: Save everything permanently
# ---------------------------------------------------------
joblib.dump(fusion_model, "./data/processed/fusion_model.pkl")
joblib.dump(scaler, "./data/processed/fusion_scaler.pkl")
print("\n✅ Saved fusion model to ./data/processed/fusion_model.pkl")
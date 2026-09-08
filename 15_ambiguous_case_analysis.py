import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# STEP 1: Load data, same as before
# ---------------------------------------------------------
manifest_df = pd.read_csv("./data/processed/fusion_ready_manifest.csv")
image_features = np.load("./data/processed/image_features.npy")
env_feature_cols = ['max_dhw_trailing_12wk', 'mean_sst_trailing_12wk', 'mean_sst_anomaly_trailing_12wk']
env_features = manifest_df[env_feature_cols].values
combined_features = np.hstack([image_features, env_features])
labels = (manifest_df['label'] == 'healthy').astype(int).values

train_mask = (manifest_df['split'] == 'train').values
test_mask = (manifest_df['split'] == 'test').values

# ---------------------------------------------------------
# STEP 2: Train image-only model, get its CONFIDENCE (not just yes/no) on test data
# ---------------------------------------------------------
scaler_img = StandardScaler()
X_train_img = scaler_img.fit_transform(image_features[train_mask])
X_test_img = scaler_img.transform(image_features[test_mask])

image_model = MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=500, random_state=42, early_stopping=True)
image_model.fit(X_train_img, labels[train_mask])
image_probs = image_model.predict_proba(X_test_img)[:, 1]  # confidence, not just 0/1
image_preds = (image_probs > 0.5).astype(int)

# ---------------------------------------------------------
# STEP 3: Identify the AMBIGUOUS cases -- where confidence is close to 50%
# WHY 0.5: a prediction of 0.5 means "I genuinely can't tell," while 0.95 or 0.05
# means the model is very sure. We're isolating the genuinely uncertain cases.
# ---------------------------------------------------------
uncertainty = np.abs(image_probs - 0.5)
ambiguous_mask_in_test = uncertainty < 0.15  # predictions between 35%-65% confidence

print(f"Out of {test_mask.sum()} test images, {ambiguous_mask_in_test.sum()} are 'ambiguous' (image model unsure).")

y_test = labels[test_mask]
image_acc_ambiguous = accuracy_score(y_test[ambiguous_mask_in_test], image_preds[ambiguous_mask_in_test])
image_acc_overall = accuracy_score(y_test, image_preds)

print(f"\nImage-only accuracy on ALL test images: {image_acc_overall*100:.2f}%")
print(f"Image-only accuracy on AMBIGUOUS images only: {image_acc_ambiguous*100:.2f}%")
print("(This should be noticeably lower -- confirming these really are the hard cases)")

# ---------------------------------------------------------
# STEP 4: Train fusion model, check its accuracy on those SAME ambiguous images
# ---------------------------------------------------------
scaler_fusion = StandardScaler()
X_train_fusion = scaler_fusion.fit_transform(combined_features[train_mask])
X_test_fusion = scaler_fusion.transform(combined_features[test_mask])

fusion_model = MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=500, random_state=42, early_stopping=True)
fusion_model.fit(X_train_fusion, labels[train_mask])
fusion_preds = fusion_model.predict(X_test_fusion)

fusion_acc_ambiguous = accuracy_score(y_test[ambiguous_mask_in_test], fusion_preds[ambiguous_mask_in_test])
fusion_acc_overall = accuracy_score(y_test, fusion_preds)

print(f"\nFusion accuracy on ALL test images: {fusion_acc_overall*100:.2f}%")
print(f"Fusion accuracy on AMBIGUOUS images only: {fusion_acc_ambiguous*100:.2f}%")

print("\n" + "="*60)
print("THE KEY COMPARISON:")
print(f"  On AMBIGUOUS (hard) cases: Image-only={image_acc_ambiguous*100:.2f}%  Fusion={fusion_acc_ambiguous*100:.2f}%")
print("="*60)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from statsmodels.stats.contingency_tables import mcnemar

manifest_df = pd.read_csv("./data/processed/fusion_ready_manifest.csv")
image_features = np.load("./data/processed/image_features.npy")
env_feature_cols = ['max_dhw_trailing_12wk', 'mean_sst_trailing_12wk', 'mean_sst_anomaly_trailing_12wk']
env_features = manifest_df[env_feature_cols].values
combined_features = np.hstack([image_features, env_features])
labels = (manifest_df['label'] == 'healthy').astype(int).values

train_mask = (manifest_df['split'] == 'train').values
test_mask = (manifest_df['split'] == 'test').values
y_test = labels[test_mask]

scaler_img = StandardScaler()
X_train_img = scaler_img.fit_transform(image_features[train_mask])
X_test_img = scaler_img.transform(image_features[test_mask])
image_model = MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=500, random_state=42, early_stopping=True)
image_model.fit(X_train_img, labels[train_mask])
image_probs = image_model.predict_proba(X_test_img)[:, 1]
image_preds = (image_probs > 0.5).astype(int)

scaler_fusion = StandardScaler()
X_train_fusion = scaler_fusion.fit_transform(combined_features[train_mask])
X_test_fusion = scaler_fusion.transform(combined_features[test_mask])
fusion_model = MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=500, random_state=42, early_stopping=True)
fusion_model.fit(X_train_fusion, labels[train_mask])
fusion_preds = fusion_model.predict(X_test_fusion)

uncertainty = np.abs(image_probs - 0.5)
ambiguous_mask = uncertainty < 0.15

y_amb = y_test[ambiguous_mask]
image_amb = image_preds[ambiguous_mask]
fusion_amb = fusion_preds[ambiguous_mask]

# McNemar's table: count cases where the two models DISAGREE
both_right = np.sum((image_amb == y_amb) & (fusion_amb == y_amb))
image_only_right = np.sum((image_amb == y_amb) & (fusion_amb != y_amb))
fusion_only_right = np.sum((image_amb != y_amb) & (fusion_amb == y_amb))
both_wrong = np.sum((image_amb != y_amb) & (fusion_amb != y_amb))

table = [[both_right, image_only_right], [fusion_only_right, both_wrong]]
print("McNemar contingency table (on the 110 ambiguous images):")
print(f"  Both correct: {both_right}, Only image-only correct: {image_only_right}")
print(f"  Only fusion correct: {fusion_only_right}, Both wrong: {both_wrong}")

result = mcnemar(table, exact=True)
print(f"\np-value: {result.pvalue:.4f}")
if result.pvalue < 0.05:
    print("✅ Statistically significant -- fusion's advantage on ambiguous cases is real, not noise.")
else:
    print("⚠️ NOT statistically significant at p<0.05 -- with only 110 images, we cannot confidently")
    print("claim fusion helps on hard cases. Report this honestly as a suggestive but inconclusive trend.")
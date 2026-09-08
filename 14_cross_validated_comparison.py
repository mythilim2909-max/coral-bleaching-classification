import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# STEP 1: Load the combined fusion-ready data
# ---------------------------------------------------------
manifest_df = pd.read_csv("./data/processed/fusion_ready_manifest.csv")
image_features = np.load("./data/processed/image_features.npy")
env_feature_cols = ['max_dhw_trailing_12wk', 'mean_sst_trailing_12wk', 'mean_sst_anomaly_trailing_12wk']
env_features = manifest_df[env_feature_cols].values

combined_features = np.hstack([image_features, env_features])
labels = (manifest_df['label'] == 'healthy').astype(int).values  # matches your corrected convention

# ---------------------------------------------------------
# STEP 2: 5-fold cross-validation -- train/test 5 DIFFERENT times on
# different random slices, so one lucky/unlucky split can't mislead us
# ---------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fusion_scores = []
image_only_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(combined_features, labels)):
    # Fusion model (image + environment)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(combined_features[train_idx])
    X_test = scaler.transform(combined_features[test_idx])
    fusion_model = MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=500, random_state=42, early_stopping=True)
    fusion_model.fit(X_train, labels[train_idx])
    fusion_acc = accuracy_score(labels[test_idx], fusion_model.predict(X_test))
    fusion_scores.append(fusion_acc)

    # Image-only model (same architecture, minus the 3 environmental columns)
    X_train_img = scaler.fit_transform(image_features[train_idx])
    X_test_img = scaler.transform(image_features[test_idx])
    image_model = MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=500, random_state=42, early_stopping=True)
    image_model.fit(X_train_img, labels[train_idx])
    image_acc = accuracy_score(labels[test_idx], image_model.predict(X_test_img))
    image_only_scores.append(image_acc)

    print(f"Fold {fold+1}: Fusion={fusion_acc*100:.2f}%  Image-only={image_acc*100:.2f}%")

print(f"\nFusion:     mean={np.mean(fusion_scores)*100:.2f}%  std={np.std(fusion_scores)*100:.2f}%")
print(f"Image-only: mean={np.mean(image_only_scores)*100:.2f}%  std={np.std(image_only_scores)*100:.2f}%")
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# STEP 1: Load the SAME 1,139-image test set used for fusion and environment-only
# ---------------------------------------------------------
manifest_df = pd.read_csv("./data/processed/fusion_ready_manifest.csv")
test_df = manifest_df[manifest_df['split'] == 'test'].reset_index(drop=True)
print(f"Re-testing EfficientNet on the exact same {len(test_df)} images used for fusion.\n")

# ---------------------------------------------------------
# STEP 2: Load the FULL EfficientNet model (not the feature extractor --
# we want its actual final healthy/bleached answer this time)
# ---------------------------------------------------------
model = tf.keras.models.load_model("./data/processed/coral_efficientnet_model.keras")

def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    return tf.keras.utils.img_to_array(img)

# ---------------------------------------------------------
# STEP 3: Run predictions in batches
# ---------------------------------------------------------
BATCH_SIZE = 32
paths = test_df['saved_path'].tolist()
true_labels = (test_df['label'] == 'healthy').astype(int).values
predictions = []
for i in range(0, len(paths), BATCH_SIZE):
    batch_paths = paths[i:i + BATCH_SIZE]
    batch_imgs = np.stack([load_image(p) for p in batch_paths])
    batch_preds = model.predict(batch_imgs, verbose=0)
    predictions.extend(batch_preds.flatten())

predicted_labels = (np.array(predictions) > 0.5).astype(int)

# ---------------------------------------------------------
# STEP 4: The real, comparable accuracy
# ---------------------------------------------------------
fair_accuracy = accuracy_score(true_labels, predicted_labels)
print(f"✅ EfficientNet accuracy on the SAME 1,139-image fusion test set: {fair_accuracy*100:.2f}%")
print("\nDetailed report:")
print(classification_report(true_labels, predicted_labels, target_names=['healthy', 'bleached']))

print("\n" + "="*60)
print("FINAL, FAIR 3-WAY COMPARISON (all tested on the same 1,139 images):")
print(f"  Environment-only:  66.73%")
print(f"  Image-only (EfficientNet, re-tested): {fair_accuracy*100:.2f}%")
print(f"  Fusion:            87.36%")
print("="*60)
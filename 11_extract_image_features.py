import tensorflow as tf
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# STEP 1: Load the fusion-ready dataset (7,591 images with real env data)
# ---------------------------------------------------------
df = pd.read_csv("./data/processed/coral_labels_with_environment.csv")
df = df.dropna(subset=['max_dhw_trailing_12wk', 'mean_sst_trailing_12wk', 'mean_sst_anomaly_trailing_12wk']).reset_index(drop=True)
print(f"Extracting image features for {len(df)} images.")
print(df['split'].value_counts())

# ---------------------------------------------------------
# STEP 2: Load your already-trained MobileNetV2 model, and cut off its
# final decision layer -- we want its "visual clues" output, not its final answer
# ---------------------------------------------------------
full_model = tf.keras.models.load_model("./data/processed/coral_efficientnet_model.keras")
feature_extractor = tf.keras.Model(inputs=full_model.input, outputs=full_model.layers[-2].output)
print(f"Feature extractor ready. Output size per image: {feature_extractor.output_shape[-1]} numbers.")

# ---------------------------------------------------------
# STEP 3: Run every image through it, in batches, with progress printed
# ---------------------------------------------------------
def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    return tf.keras.utils.img_to_array(img)

BATCH_SIZE = 32
paths = df['saved_path'].tolist()
all_features = []

print("\nThis will take a while on CPU -- progress prints every ~640 images.\n")
for i in range(0, len(paths), BATCH_SIZE):
    batch_paths = paths[i:i + BATCH_SIZE]
    batch_imgs = np.stack([load_image(p) for p in batch_paths])
    batch_features = feature_extractor.predict(batch_imgs, verbose=0)
    all_features.append(batch_features)

    if (i // BATCH_SIZE) % 20 == 0:
        print(f"  ...processed {i + len(batch_paths)} of {len(paths)} images")

image_features = np.vstack(all_features)
print(f"\n✅ Done. Feature matrix shape: {image_features.shape}")

# ---------------------------------------------------------
# STEP 4: Save features + matching manifest so we NEVER have to redo this slow step
# ---------------------------------------------------------
np.save("./data/processed/image_features.npy", image_features)
df.to_csv("./data/processed/fusion_ready_manifest.csv", index=False)
print("Saved to ./data/processed/image_features.npy and fusion_ready_manifest.csv")
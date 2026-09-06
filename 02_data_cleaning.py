import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Load the manifest we just built (real, verified labels)
manifest_df = pd.read_csv("./data/processed/coral_labels_manifest.csv")
print(f"Loaded manifest with {len(manifest_df)} labeled images.")

# STEP 1: Check for corrupted images
print("\nChecking for corrupted images...")
def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False

manifest_df['is_valid'] = manifest_df['filepath'].apply(is_valid_image)
broken_count = (~manifest_df['is_valid']).sum()
print(f"Broken images found: {broken_count}")
manifest_df = manifest_df[manifest_df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)
print(f"Remaining valid images: {len(manifest_df)}")

# STEP 2: Try to recover the ORIGINAL train/validation/test split from the file path
def get_split_from_path(path):
    path_lower = path.lower()
    if "validation" in path_lower or "valid" in path_lower:
        return "validation"
    elif "test" in path_lower:
        return "test"
    elif "train" in path_lower:
        return "train"
    else:
        return None

manifest_df['split'] = manifest_df['filepath'].apply(get_split_from_path)
found_splits = manifest_df['split'].notna().sum()
print(f"\nOriginal split recovered for {found_splits} out of {len(manifest_df)} images.")

if manifest_df['split'].isna().any():
    print("Could not fully recover original splits -- creating our own 70/15/15 split instead.")
    train_df, temp_df = train_test_split(
        manifest_df, test_size=0.30, stratify=manifest_df['label'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42
    )
    train_df = train_df.copy(); train_df['split'] = 'train'
    val_df = val_df.copy(); val_df['split'] = 'validation'
    test_df = test_df.copy(); test_df['split'] = 'test'
    manifest_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print("\nFinal split sizes:")
print(manifest_df['split'].value_counts())
print("\nClass balance per split:")
print(manifest_df.groupby(['split', 'label']).size())

# STEP 3: Quick check -- are images actually 224x224?
print("\nChecking image sizes on a sample of 500...")
wrong_size = 0
for path in manifest_df['filepath'].sample(min(500, len(manifest_df)), random_state=1):
    if Image.open(path).size != (224, 224):
        wrong_size += 1
print(f"Out of 500 checked, {wrong_size} were not 224x224 (these will be resized automatically below).")

# STEP 4: Save clean, organized copies into split/label folders
out_base = "./data/processed/images"
os.makedirs(out_base, exist_ok=True)

saved = 0
skipped = 0
for idx, row in manifest_df.iterrows():
    out_dir = os.path.join(out_base, row['split'], row['label'])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"img_{idx}.jpg")

    if os.path.exists(out_path):
        skipped += 1
        continue

    img = Image.open(row['filepath']).convert('RGB')
    if img.size != (224, 224):
        img = img.resize((224, 224))
    img.save(out_path, quality=95)
    saved += 1

    if saved % 1000 == 0:
        print(f"  ...saved {saved} images so far")

print(f"\nDone saving. {saved} new images saved, {skipped} already existed and were skipped.")

# STEP 5: Save the FINAL clean manifest -- our single source of truth from now on
manifest_df['saved_path'] = [
    os.path.join(out_base, row['split'], row['label'], f"img_{idx}.jpg")
    for idx, row in manifest_df.iterrows()
]
final_csv = "./data/processed/coral_labels_clean_final.csv"
manifest_df[['split', 'label', 'saved_path']].to_csv(final_csv, index=False)

print(f"\n✅ CLEANING COMPLETE.")
print(f"Final manifest saved to: {final_csv}")
print(f"Cleaned images saved to: {out_base}")   
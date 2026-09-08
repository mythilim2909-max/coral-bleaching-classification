import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

print("Reconstructing dataset pipeline to recover original filenames...")

# 1. Load the original raw manifest (which has the original filepaths)
manifest_df = pd.read_csv("./data/processed/coral_labels_manifest.csv")

# 2. Quickly repeat the exact cleaning split so the rows match our final images perfectly
def is_valid_image(path):
    try:
        Image.open(path).verify()
        return True
    except Exception:
        return False

manifest_df['is_valid'] = manifest_df['filepath'].apply(is_valid_image)
manifest_df = manifest_df[manifest_df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)

def get_split_from_path(path):
    path_lower = path.lower()
    if "validation" in path_lower or "valid" in path_lower: return "validation"
    elif "test" in path_lower: return "test"
    elif "train" in path_lower: return "train"
    return None

manifest_df['split'] = manifest_df['filepath'].apply(get_split_from_path)
if manifest_df['split'].isna().any():
    train_df, temp_df = train_test_split(manifest_df, test_size=0.30, stratify=manifest_df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)
    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()
    train_df['split'], val_df['split'], test_df['split'] = 'train', 'validation', 'test'
    manifest_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# 3. Extract the Site ID and Year from the original filepath
def extract_metadata(filepath):
    filename = os.path.basename(filepath) # e.g., OAH-B3277_2019_24_43054.PNG
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1] # Returns Site ID, Year
    return "UNKNOWN", "UNKNOWN"

print("Extracting Site IDs and Years...")
manifest_df['site_id'], manifest_df['year'] = zip(*manifest_df['filepath'].apply(extract_metadata))

# 4. Attach this new data to our final clean CSV
final_df = pd.read_csv("./data/processed/coral_labels_clean_final.csv")
final_df['site_id'] = manifest_df['site_id']
final_df['year'] = manifest_df['year']

# Save the updated master spreadsheet
final_df.to_csv("./data/processed/coral_labels_clean_final.csv", index=False)

print("\n✅ Metadata Extraction Complete!")
print(f"Total Unique Coral Survey Sites Found: {final_df['site_id'].nunique()}")
print("\nSample of updated data:")
print(final_df[['split', 'label', 'site_id', 'year']].head())
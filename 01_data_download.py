from huggingface_hub import snapshot_download
import os
import shutil
from PIL import Image
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# STEP 1: Download the REAL original files (not the auto-converted preview)
# ---------------------------------------------------------
print("Downloading the actual repository files...")
local_path = snapshot_download(
    repo_id="NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset",
    repo_type="dataset",
    cache_dir="./data/raw/hf_cache"
)
print("Downloaded to:", local_path)

# ---------------------------------------------------------
# STEP 2: PROOF CHECK — print the real folder structure so we can SEE it
# ---------------------------------------------------------
print("\nActual folder structure found:")
found_label_folders = set()
for root, dirs, files in os.walk(local_path):
    for d in dirs:
        if d.upper() in ["CORAL", "CORAL_BL"]:
            found_label_folders.add(d)
    # only print first couple levels so it's not overwhelming
    depth = root.replace(local_path, '').count(os.sep)
    if depth <= 3:
        print("  " * depth + os.path.basename(root) + "/")

print("\nLabel folders detected:", found_label_folders)

if not found_label_folders:
    print("\n⚠️ STOP: No CORAL / CORAL_BL folders found in the raw files either.")
    print("Paste me this entire printed folder structure above — do not proceed further.")
else:
    print("\n✅ Found real label folders. Proceeding to build the labeled dataset.")

    # ---------------------------------------------------------
    # STEP 3: Walk every image file and record its TRUE label from its real folder name
    # ---------------------------------------------------------
    rows = []
    for root, dirs, files in os.walk(local_path):
        folder_name = os.path.basename(root).upper()
        if folder_name in ["CORAL", "CORAL_BL"]:
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, fname)
                    label = "healthy" if folder_name == "CORAL" else "bleached"
                    rows.append({"filepath": full_path, "label": label})

    manifest_df = pd.DataFrame(rows)
    print(f"\nTotal labeled images found: {len(manifest_df)}")
    print("Class balance:", Counter(manifest_df["label"]))

    if len(manifest_df) == 0:
        print("\n⚠️ STOP: Folders were found but contained no images. Paste this output to me.")
    else:
        # ---------------------------------------------------------
        # STEP 4: Visual sanity check with REAL labels
        # ---------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        sample_rows = manifest_df.sample(6, random_state=1).reset_index(drop=True)
        for i, ax in enumerate(axes.flat):
            img = Image.open(sample_rows.loc[i, "filepath"])
            ax.imshow(img)
            ax.set_title(f"Label: {sample_rows.loc[i, 'label']}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()

        # ---------------------------------------------------------
        # STEP 5: Save this manifest — this is our source of truth going forward
        # ---------------------------------------------------------
        os.makedirs("./data/processed", exist_ok=True)
        manifest_df.to_csv("./data/processed/coral_labels_manifest.csv", index=False)
        print("\n✅ Saved manifest to ./data/processed/coral_labels_manifest.csv")
        print("This file has the REAL, verified label for every image, based on its actual folder.")
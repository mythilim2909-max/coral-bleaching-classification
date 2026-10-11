import pandas as pd

# Your existing manifest (already has real site_id, year, extracted earlier)
manifest_df = pd.read_csv("./data/processed/coral_labels_clean_final.csv")

# The real NOAA GPS metadata you just downloaded
gps_df = pd.read_csv("./data/raw/NOAA_ESD_CoralBleachingClassifier_ImageMetadata.csv")
gps_df = gps_df.rename(columns={"Site": "site_id", "Latitude": "latitude", "Longitude": "longitude", "Date": "survey_date"})
gps_df_unique_sites = gps_df[["site_id", "latitude", "longitude"]].drop_duplicates(subset="site_id")

# Merge on site_id
merged_df = manifest_df.merge(gps_df_unique_sites, on="site_id", how="left")

matched = merged_df["latitude"].notna().sum()
total = len(merged_df)
print(f"Matched real GPS coordinates for {matched} out of {total} images ({matched/total*100:.1f}%)")
print()
print("Sites that did NOT match (need honest handling):")
unmatched_sites = merged_df[merged_df["latitude"].isna()]["site_id"].unique()
print(f"{len(unmatched_sites)} unique unmatched sites, e.g.: {unmatched_sites[:10]}")

merged_df.to_csv("./data/processed/coral_labels_with_real_gps.csv", index=False)
print("\nSaved to coral_labels_with_real_gps.csv")
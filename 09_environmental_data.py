import pandas as pd
import numpy as np
import requests
from io import StringIO
import time
from datetime import timedelta

# ---------------------------------------------------------
# STEP 0: Load GPS-matched data, recover survey_date
# ---------------------------------------------------------
df = pd.read_csv("./data/processed/coral_labels_with_real_gps.csv")
matched_df = df[df['latitude'].notna()].copy()
print(f"Loaded {len(df)} total images, {len(matched_df)} with real GPS matches.")

gps_source_df = pd.read_csv("./data/raw/NOAA_ESD_CoralBleachingClassifier_ImageMetadata.csv")
gps_source_df = gps_source_df.rename(columns={"Site": "site_id", "Date": "survey_date"})
site_dates = gps_source_df[["site_id", "survey_date"]].drop_duplicates(subset="site_id")
matched_df = matched_df.merge(site_dates, on="site_id", how="left")
print(f"Recovered survey_date for {matched_df['survey_date'].notna().sum()} out of {len(matched_df)} images.")

# ---------------------------------------------------------
# STEP 1: One row per unique site
# ---------------------------------------------------------
unique_sites = matched_df[['site_id', 'latitude', 'longitude', 'survey_date']].drop_duplicates(subset='site_id').reset_index(drop=True)
print(f"Will query NOAA for {len(unique_sites)} unique real site locations.\n")

# ---------------------------------------------------------
# STEP 2: Query the CONFIRMED working server, one variable at a time
# ---------------------------------------------------------
BASE_URL = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.csv"
VARIABLES = ["CRW_DHW", "CRW_SST", "CRW_SSTANOMALY"]

def fetch_one_variable(var_name, lat, lon, start_str, end_str):
    url = f"{BASE_URL}?{var_name}[({start_str}):1:({end_str})][({lat})][({lon})]"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text), skiprows=[1])
    return data[var_name]

def fetch_environmental_history(lat, lon, survey_date_str, lookback_days=84):
    try:
        end_date = pd.to_datetime(survey_date_str)
        start_date = end_date - timedelta(days=lookback_days)
        start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_str = end_date.strftime("%Y-%m-%dT00:00:00Z")

        dhw_series = fetch_one_variable("CRW_DHW", lat, lon, start_str, end_str)
        time.sleep(0.3)
        sst_series = fetch_one_variable("CRW_SST", lat, lon, start_str, end_str)
        time.sleep(0.3)
        ssta_series = fetch_one_variable("CRW_SSTANOMALY", lat, lon, start_str, end_str)
        time.sleep(0.3)

        return {
            'max_dhw_trailing_12wk': dhw_series.max(),
            'mean_sst_trailing_12wk': sst_series.mean(),
            'mean_sst_anomaly_trailing_12wk': ssta_series.mean(),
            'query_success': True
        }
    except Exception as e:
        print(f"  Failed for lat={lat}, lon={lon}, date={survey_date_str}: {e}")
        return {
            'max_dhw_trailing_12wk': np.nan,
            'mean_sst_trailing_12wk': np.nan,
            'mean_sst_anomaly_trailing_12wk': np.nan,
            'query_success': False
        }

print("Querying real ocean temperature data (this will take a few minutes)...")
results = []
for i, row in unique_sites.iterrows():
    env_data = fetch_environmental_history(row['latitude'], row['longitude'], row['survey_date'])
    env_data['site_id'] = row['site_id']
    results.append(env_data)

    if (i + 1) % 10 == 0:
        print(f"  ...processed {i + 1} of {len(unique_sites)} sites")

env_results_df = pd.DataFrame(results)
success_count = env_results_df['query_success'].sum()
print(f"\nSuccessfully retrieved real environmental data for {success_count} out of {len(unique_sites)} sites.")

# ---------------------------------------------------------
# STEP 3: Merge back onto every image
# ---------------------------------------------------------
final_df = matched_df.merge(env_results_df, on='site_id', how='left')
final_df.to_csv("./data/processed/coral_labels_with_environment.csv", index=False)

print(f"\n✅ Saved {len(final_df)} images with real GPS + real environmental data")
print("to: ./data/processed/coral_labels_with_environment.csv")
print("\nSample of final environmental features:")
print(final_df[['site_id', 'max_dhw_trailing_12wk', 'mean_sst_trailing_12wk', 'mean_sst_anomaly_trailing_12wk']].drop_duplicates(subset='site_id').head(10))
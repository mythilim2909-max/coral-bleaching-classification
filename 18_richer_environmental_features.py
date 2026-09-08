import pandas as pd
import numpy as np
import requests
from io import StringIO
import time
from datetime import timedelta

# ---------------------------------------------------------
# STEP 1: Load the same real sites as before
# ---------------------------------------------------------
df = pd.read_csv("./data/processed/coral_labels_with_real_gps.csv")
matched_df = df[df['latitude'].notna()].copy()

gps_source_df = pd.read_csv("./data/raw/NOAA_ESD_CoralBleachingClassifier_ImageMetadata.csv")
gps_source_df = gps_source_df.rename(columns={"Site": "site_id", "Date": "survey_date"})
site_dates = gps_source_df[["site_id", "survey_date"]].drop_duplicates(subset="site_id")
matched_df = matched_df.merge(site_dates, on="site_id", how="left")

unique_sites = matched_df[['site_id', 'latitude', 'longitude', 'survey_date']].drop_duplicates(subset='site_id').reset_index(drop=True)
print(f"Pulling richer environmental data for {len(unique_sites)} unique real sites.\n")

# ---------------------------------------------------------
# STEP 2: Same confirmed-working server, now pulling 5 variables instead of 3
# ---------------------------------------------------------
BASE_URL = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.csv"
VARIABLES = ["CRW_DHW", "CRW_SST", "CRW_SSTANOMALY", "CRW_HOTSPOT", "CRW_BAA"]

def fetch_one_variable(var_name, lat, lon, start_str, end_str):
    url = f"{BASE_URL}?{var_name}[({start_str}):1:({end_str})][({lat})][({lon})]"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text), skiprows=[1])
    return data[var_name]

def compute_rich_features(lat, lon, survey_date_str, lookback_days=84):
    try:
        end_date = pd.to_datetime(survey_date_str)
        start_date = end_date - timedelta(days=lookback_days)
        start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_str = end_date.strftime("%Y-%m-%dT00:00:00Z")

        series = {}
        for var in VARIABLES:
            series[var] = fetch_one_variable(var, lat, lon, start_str, end_str)
            time.sleep(0.3)

        dhw = series["CRW_DHW"]
        sst = series["CRW_SST"]
        ssta = series["CRW_SSTANOMALY"]
        hotspot = series["CRW_HOTSPOT"]
        baa = series["CRW_BAA"]

        # Trend: compare the last 2 weeks' average DHW to the first 2 weeks' -- is it accelerating?
        first_2wk_mean = dhw.iloc[:14].mean()
        last_2wk_mean = dhw.iloc[-14:].mean()
        dhw_trend = last_2wk_mean - first_2wk_mean

        return {
            'max_dhw_trailing_12wk': dhw.max(),
            'mean_dhw_trailing_12wk': dhw.mean(),
            'days_dhw_gt_4': (dhw > 4).sum(),
            'days_dhw_gt_8': (dhw > 8).sum(),
            'mean_sst_trailing_12wk': sst.mean(),
            'max_sst_trailing_12wk': sst.max(),
            'std_sst_trailing_12wk': sst.std(),
            'mean_sst_anomaly_trailing_12wk': ssta.mean(),
            'max_sst_anomaly_trailing_12wk': ssta.max(),
            'mean_hotspot_trailing_12wk': hotspot.mean(),
            'max_hotspot_trailing_12wk': hotspot.max(),
            'max_baa_trailing_12wk': baa.max(),
            'dhw_trend': dhw_trend,
            'query_success': True
        }
    except Exception as e:
        print(f"  Failed for lat={lat}, lon={lon}, date={survey_date_str}: {e}")
        return {k: np.nan for k in [
            'max_dhw_trailing_12wk', 'mean_dhw_trailing_12wk', 'days_dhw_gt_4', 'days_dhw_gt_8',
            'mean_sst_trailing_12wk', 'max_sst_trailing_12wk', 'std_sst_trailing_12wk',
            'mean_sst_anomaly_trailing_12wk', 'max_sst_anomaly_trailing_12wk',
            'mean_hotspot_trailing_12wk', 'max_hotspot_trailing_12wk', 'max_baa_trailing_12wk',
            'dhw_trend']} | {'query_success': False}

print("Querying NOAA for richer data (5 variables x 127 sites -- will take longer than before)...")
results = []
for i, row in unique_sites.iterrows():
    feats = compute_rich_features(row['latitude'], row['longitude'], row['survey_date'])
    feats['site_id'] = row['site_id']
    results.append(feats)
    if (i + 1) % 10 == 0:
        print(f"  ...processed {i + 1} of {len(unique_sites)} sites")

env_results_df = pd.DataFrame(results)
print(f"\nSuccessfully retrieved rich data for {env_results_df['query_success'].sum()} out of {len(unique_sites)} sites.")

final_df = matched_df.merge(env_results_df, on='site_id', how='left')
final_df.to_csv("./data/processed/coral_labels_rich_environment.csv", index=False)
print("\n✅ Saved to ./data/processed/coral_labels_rich_environment.csv")
print("\nSample of new features:")
print(final_df[['site_id', 'max_dhw_trailing_12wk', 'days_dhw_gt_4', 'max_baa_trailing_12wk', 'dhw_trend']].drop_duplicates(subset='site_id').head(10))
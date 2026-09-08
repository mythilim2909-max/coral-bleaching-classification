import pandas as pd
final_df = pd.read_csv("./data/processed/coral_labels_with_environment.csv")

total = len(final_df)
missing_env = final_df['max_dhw_trailing_12wk'].isna().sum()
usable = total - missing_env

print(f"Total images with GPS: {total}")
print(f"Images missing environmental data (land-masked pixels): {missing_env}")
print(f"Final usable images for fusion model: {usable}")
print(f"\nUnique sites affected: {final_df[final_df['max_dhw_trailing_12wk'].isna()]['site_id'].unique()}")

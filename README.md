# Coral Bleaching Detection with Environmental Context

Does adding the sea-temperature history of a reef site improve coral-bleaching classification from photographs?

This project compares three setups on real NOAA reef imagery from Hawaii:

1. **Image only**: a CNN / EfficientNetB0 looks at the coral photo.
2. **Environment only**: a Random Forest uses NOAA satellite temperature features for the 12 weeks before the photo.
3. **Fusion**: image features and temperature features are combined in one classifier.

A Streamlit app (`app.py`) shows the results and lets you try the models on a coral image.

---

## Main findings

All figures are from one training run. Neural-network training is random, so rerunning the scripts changes the numbers slightly (see [Reproducibility](#reproducibility)).

| Model | Test accuracy | Notes |
|---|---|---|
| Baseline CNN (from scratch) | 83.17% | Full test set, 1,563 images |
| MobileNetV2 (transfer learning) | 84.39% | Same test set |
| EfficientNetB0 (2-phase fine-tuning) | 87.52% | Same test set |
| Environment only (Random Forest, 3 features) | 66.73% | Always guessing the majority class scores 64.62% |
| Image only: EfficientNetB0 end-to-end | 87.45% | Same 1,139-image test set as fusion |
| Image only: EfficientNet features + MLP | 86.48% | Same classifier as fusion, minus the 3 temperature numbers |
| Image + environment: fusion MLP | 87.36% | Within noise of image-only |

Additional analysis:

- **5-fold cross-validation:** fusion 86.73% ± 0.54 vs image-only 87.22% ± 0.66.
- **106 ambiguous test images** (where the image model was least confident): image-only 56.60% vs fusion 72.64%. McNemar's test gave p = 0.006 in this run. Earlier runs gave p = 0.65 and p = 0.15, so this result is not stable across runs.
- **13 environmental features instead of 3** (including NOAA's Bleaching Alert Area) did not help: 66.67% vs 66.73%.

**Overall:** temperature data alone carries a modest signal, but combining it with the image model gives about the same overall accuracy as the image model alone.

---

## Data

| Source | What it provides |
|---|---|
| [NOAA PIFSC ESD Coral Bleaching Dataset](https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset) (Hugging Face) | Coral close-up images labelled healthy or bleached |
| NOAA ESD image metadata CSV | Site IDs, GPS coordinates, survey dates |
| NOAA Coral Reef Watch 5 km daily product (via the PacIOOS ERDDAP server) | Degree Heating Weeks, sea surface temperature, SST anomaly |

Environmental features per site, over the 84 days (12 weeks) before the survey date:

- maximum Degree Heating Weeks
- mean sea surface temperature
- mean SST anomaly

Final dataset: **7,591 images** with image + GPS + temperature data. Images at near-shore (land-masked) satellite pixels have no temperature data and are excluded.

### What is and is not in this repository

- **In the repo:** all scripts, `app.py`, the processed CSV files, the saved feature file (`image_features.npy`) and the saved MobileNetV2 / EfficientNetB0 / environment / fusion models.
- **Not in the repo:** the raw downloaded dataset, and the baseline CNN model (`coral_baseline_model.keras`, about 137 MB, over GitHub's 100 MB limit). Recreate it with `03_model_training.py`.
- **In the Releases section:** `images.zip`, the cleaned and resized 224x224 images, organised as `train/`, `validation/` and `test/` folders, each with `healthy/` and `bleached/` subfolders.

---

## Setup (Windows, PowerShell)

Tested with Python 3.12.

1. Download or clone this repository, and open the folder in VS Code (**File → Open Folder...**).
2. Open a terminal (**Terminal → New Terminal**) and create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   If PowerShell says running scripts is disabled, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again.
3. Install the libraries:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Download `images.zip` from the **Releases** page, extract it, and place the folders so you have:
   ```
   data/processed/images/train/...
   data/processed/images/validation/...
   data/processed/images/test/...
   ```

Run all commands from the project root, the folder that contains `data/`. Every script uses paths like `./data/processed/...`.

---

## Scripts

| # | Script | What it does |
|---|---|---|
| 01 | `01_data_download.py` | Downloads the raw dataset from Hugging Face and builds a label manifest from folder names |
| 02 | `02_data_cleaning.py` | Removes corrupt images, creates train/validation/test splits, resizes to 224x224 |
| 03 | `03_model_training.py` | Trains the baseline CNN |
| 04 | `04_model_evaluation.py` | Evaluates the baseline CNN (confusion matrix) |
| 05 | `05_transfer_learning.py` | Trains MobileNetV2 with transfer learning |
| 06 | `06_efficientnet_training.py` | Trains EfficientNetB0 (warm-up, then fine-tuning) |
| 07 | `07_site_metadata_extraction.py` | Extracts site ID and year from the original filenames |
| 08 | `08_merge_gps_coordinates.py` | Merges GPS coordinates onto each image by site ID |
| 09 | `09_environmental_data.py` | Downloads NOAA temperature history for each site |
| 10 | `10_environmental_only_baseline.py` | Environment-only Random Forest |
| 11 | `11_extract_image_features.py` | Extracts 1,280 EfficientNet features per image |
| 12 | `12_fusion_model.py` | Trains the image + environment fusion MLP |
| 13 | `13_fair_comparison_eval.py` | Re-tests EfficientNet on the same images as fusion |
| 14 | `14_cross_validated_comparison.py` | 5-fold cross-validation: fusion vs image-only |
| 15 | `15_ambiguous_case_analysis.py` | Accuracy on the images the image model is least sure about |
| 16 | `16_significance_test.py` | McNemar's test on those ambiguous images |
| 17 | `17_environment_model_diagnostics.py` | Checks the environment model for under/overfitting; compares Logistic Regression, SVM |
| 18 | `18_richer_environmental_features.py` | Downloads 13 richer environmental features |
| 19 | `19_richer_environment_test.py` | Tests whether the richer features help |

Helper scripts: `check_environment_coverage.py` (counts images with missing temperature data) and `test_single_variable.py` (tests one NOAA request).

---

## How to run

### Option A: Reproduce the analysis from the files in this repo

Needs the saved files in `data/processed/` and the extracted images. No raw data is required.

```powershell
python 10_environmental_only_baseline.py
python 17_environment_model_diagnostics.py
python 19_richer_environment_test.py
python 12_fusion_model.py
python 14_cross_validated_comparison.py
python 15_ambiguous_case_analysis.py
python 16_significance_test.py
python 13_fair_comparison_eval.py
```

Scripts 01, 02, 07, 08, 09 and 18 need the raw dataset and an internet connection. Scripts 03, 05 and 06 retrain models and overwrite the saved ones, so only run them if you want to retrain.

### Option B: Rebuild everything from scratch

Run scripts 01 to 19 in numeric order. Script 06 is the slowest on CPU (roughly an hour). Script 11 is also slow. Scripts 08, 09 and 18 need `data/raw/NOAA_ESD_CoralBleachingClassifier_ImageMetadata.csv`.

### Run the app

```powershell
pip install streamlit
streamlit run app.py
```

The app has four tabs: problem statement, data and methodology, results, and a demo. The demo classifies either a random held-out test image (using its saved temperature values) or an uploaded photo with temperature values you enter by hand. It needs `coral_efficientnet_model.keras`, `image_features.npy`, `fusion_ready_manifest.csv` and the images folder.

---

## Reproducibility

- The `.keras` models are trained with random initialisation, data shuffling and augmentation. **Each retraining run gives slightly different accuracy** (about one percentage point).
- Results in the table above come from one specific run. If you retrain, rerun scripts 11 to 16 afterwards, because they depend on the saved EfficientNet model, and update the numbers typed into `app.py`.
- The scikit-learn models use `random_state=42`, so they are repeatable given the same input features.
- Results quoted in `app.py` and scripts 13, 15 and 16 contain some typed-in numbers and text, so check them after a rerun.

---

## Limitations

- Trained on Hawaiian reef close-ups only. Not for conservation decisions.
- Temperature data is site-level at 5 km resolution, so all images from one site share the same values. There are only about 60 distinct environmental value combinations, so the environment model learns from far fewer distinct examples than the image count suggests.
- Only 7,591 of 10,419 images have usable environmental data.
- The ambiguous-case analysis uses a small subset (106 images) selected by the image model's own confidence, and its significance result varied between runs.
- Visible bleaching may already reflect the heat stress that preceded it, which could explain why temperature adds little. This explanation was not tested directly.

---

## Credits

- Image data: NOAA Pacific Islands Fisheries Science Center, Ecosystem Sciences Division.
- Sea-temperature data: NOAA Coral Reef Watch, served by PacIOOS.
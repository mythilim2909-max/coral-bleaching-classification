"""
Coral Bleaching Detection with Environmental Context -- Streamlit demo

HOW TO RUN (from the project root, i.e. the folder that contains the 'data' folder):
    venv\\Scripts\\activate
    pip install streamlit
    streamlit run app.py

Needs these files (already produced by your scripts):
    data/processed/coral_efficientnet_model.keras   (script 06)
    data/processed/image_features.npy               (script 11)
    data/processed/fusion_ready_manifest.csv        (script 11)
    data/processed/images/...                       (script 02)
"""
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
DATA_DIR = "./data/processed"
MODEL_PATH = f"{DATA_DIR}/coral_efficientnet_model.keras"
FEATURES_PATH = f"{DATA_DIR}/image_features.npy"
MANIFEST_PATH = f"{DATA_DIR}/fusion_ready_manifest.csv"

# Same order as script 12: image features first, then these three
ENV_COLS = [
    "max_dhw_trailing_12wk",
    "mean_sst_trailing_12wk",
    "mean_sst_anomaly_trailing_12wk",
]

# ----------------------------------------------------------------------------
# RESULTS typed in from our run logs -- EDIT THESE if you re-run and numbers change
# ----------------------------------------------------------------------------
IMAGE_MODEL_ROWS = [
    ("Baseline CNN (from scratch)", "83.17%", "Full held-out test set, 1,563 images"),
    ("MobileNetV2 (transfer learning)", "84.39%", "Same test set"),
    ("EfficientNetB0 (2-phase fine-tuning)", "87.52%", "Same test set (earlier run)"),
]
FUSION_ROWS = [
    ("Environment only (Random Forest, 3 features)", "66.73%", "Majority-class guess would score 64.62%"),
    ("Image only: EfficientNetB0 end-to-end", "87.45%", "Re-run on this machine"),
    ("Image only: EfficientNet features + MLP", "86.48%", "Same classifier as fusion, minus the 3 temperature numbers"),
    ("Image + environment: fusion MLP", "87.36%", "Difference vs image-only is within noise"),
]
AMBIGUOUS_TEXT = (
    "On the 106 test images where the image model was least confident: image-only 56.60% vs "
    "fusion 72.64%. McNemar's test gives p = 0.006, so this is statistically significant in this run "
    "(the first run gave 55.45% vs 59.09%, p = 0.65)."
)
CV_TEXT = "5-fold cross-validation: fusion 86.73% +/- 0.54 vs image-only 87.22% +/- 0.66."
FEATURE_TEXT = "Expanding the environmental features from 3 to 13 (including NOAA's Bleaching Alert Area) did not help: 66.67% vs 66.73%."

# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_data
def load_manifest():
    return pd.read_csv(MANIFEST_PATH)


@st.cache_resource(show_spinner="Loading the EfficientNet model (first time only)...")
def load_keras_models():
    import tensorflow as tf  # imported here so the other tabs still work if TensorFlow is slow/missing

    full_model = tf.keras.models.load_model(MODEL_PATH)
    # Same trick as script 11: stop one layer before the final decision -> 1,280 numbers per image
    extractor = tf.keras.Model(inputs=full_model.input, outputs=full_model.layers[-2].output)
    return full_model, extractor


@st.cache_resource(show_spinner="Preparing the fusion classifier (one time, about 20 seconds)...")
def build_fusion():
    manifest = load_manifest()
    feats = np.load(FEATURES_PATH)
    assert len(manifest) == feats.shape[0], "manifest and image_features.npy are out of sync"

    X = np.hstack([feats, manifest[ENV_COLS].values])
    y = (manifest["label"] == "healthy").astype(int).values  # 1 = healthy, same convention as scripts 14-16
    train = (manifest["split"] == "train").values
    test = (manifest["split"] == "test").values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train])
    X_test = scaler.transform(X[test])

    clf = MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=500, random_state=42, early_stopping=True)
    clf.fit(X_train, y[train])
    acc = accuracy_score(y[test], clf.predict(X_test))
    return {"model": clf, "scaler": scaler, "test_acc": float(acc), "n_test": int(test.sum())}


# ----------------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------------
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    return np.expand_dims(np.array(img, dtype="float32"), 0)  # raw 0-255, as in training


def predict_all(pil_img, env_vals, full_model, extractor, fusion):
    x = preprocess(pil_img)
    # Keras sorted the folders alphabetically: bleached=0, healthy=1, so the sigmoid output is P(healthy)
    p_healthy = float(full_model.predict(x, verbose=0)[0][0])
    p_bleached_image = 1.0 - p_healthy

    feats = extractor.predict(x, verbose=0)
    combined = np.hstack([feats, np.array([env_vals], dtype="float32")])
    scaled = fusion["scaler"].transform(combined)
    p_bleached_fusion = float(fusion["model"].predict_proba(scaled)[0][0])  # class 0 = bleached (class 1 = healthy)
    return p_bleached_image, p_bleached_fusion


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Coral Bleaching: Image + Environment", layout="wide")
st.title("Coral Bleaching Detection with Environmental Context")
st.caption("Real NOAA reef imagery + NOAA Coral Reef Watch sea-temperature data | Research prototype")

tab_overview, tab_data, tab_results, tab_demo = st.tabs(
    ["Problem statement", "Data & methodology", "Results", "Demo"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    st.header("Problem statement")
    st.write(
        "Many coral-bleaching classifiers use only the photograph of the coral. "
        "This project investigates whether adding the **sea-temperature history of the reef site** "
        "(the 12 weeks before the photo) improves the classification. Three setups are compared: "
        "image only, temperature only, and both combined."
    )
    st.header("Summary of results")
    st.write(
        "Temperature data alone gave a modest signal (66.7%, against a 64.6% majority-class baseline). "
        "Combining it with the image model gave accuracy similar to the image model alone (about 87%), "
        "across repeated splits and a significance test. One possible explanation is that visible bleaching "
        "already reflects the heat stress that preceded it, and that site-level satellite data cannot capture "
        "differences between neighbouring colonies. This explanation is a hypothesis and was not tested directly."
    )
    st.header("Work done")
    st.markdown(
        "- A full data pipeline on real NOAA data (download, clean, recover GPS, call NOAA's satellite API)\n"
        "- Three image classifiers compared (baseline CNN, MobileNetV2, EfficientNetB0)\n"
        "- A temperature-only model and an image + temperature **fusion** model\n"
        "- A statistical evaluation (cross-validation, McNemar test, feature-richness check)"
    )

# ---------------------------------------------------------------- Data & method
with tab_data:
    st.header("Data sources")
    sources = pd.DataFrame(
        [
            (
                "Coral images (10,419 labelled 224x224 patches, Hawaii 2014-2019)",
                "NOAA Pacific Islands Fisheries Science Center, Ecosystem Sciences Division. "
                "NCEI Accession 0269246. ML-ready copy: Hugging Face 'NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset'",
                "File download",
            ),
            (
                "Site GPS coordinates and survey dates",
                "NOAA_ESD_CoralBleachingClassifier_ImageMetadata.csv, same NCEI accession",
                "File download; matched to 9,159 of 10,419 images (87.9%) by site ID",
            ),
            (
                "Sea-temperature history (Degree Heating Weeks, SST, SST anomaly)",
                "NOAA Coral Reef Watch 5 km daily satellite products, served by PacIOOS ERDDAP",
                "Live API requests: 84 days before each survey date, per site",
            ),
        ],
        columns=["Component", "Source", "How we accessed it"],
    ).set_index("Component")
    st.table(sources)
    st.caption("All three sources are publicly available NOAA datasets; no synthetic data were used.")

    st.header("Methodology")
    st.markdown(
        "1. **Clean**: 0 corrupted images, stratified train/validation/test split (about 70/15/15)\n"
        "2. **Recover metadata**: site IDs from original filenames, GPS from NOAA's own metadata file\n"
        "3. **Environmental features**: max Degree Heating Weeks, mean SST and mean SST anomaly over the "
        "12 weeks before each survey (127 sites; 15 near-shore Maui sites had no valid satellite pixel)\n"
        "4. **Final dataset**: 7,591 images with image + GPS + temperature\n"
        "5. **Image model**: EfficientNetB0, transfer learning, 2-phase fine-tuning with augmentation\n"
        "6. **Fusion**: EfficientNet's 1,280-number image embedding + 3 temperature numbers, standardised, "
        "classified by a small MLP (64 and 16 neurons). This is intermediate (feature-level) fusion, not stacking."
    )
    st.header("Model selection")
    st.write(
        "MobileNetV2 and EfficientNet are compact CNNs that have been used in earlier coral-bleaching "
        "classification studies and are suitable for field deployment. YOLO-style detectors were not used "
        "because the dataset consists of pre-cropped single-subject patches, so object localisation was not required."
    )

# ---------------------------------------------------------------- Results
with tab_results:
    st.header("Image-only models")
    st.table(pd.DataFrame(IMAGE_MODEL_ROWS, columns=["Model", "Test accuracy", "Note"]).set_index("Model"))
    st.caption(
        "Training involves random initialisation, so repeated runs differ by about one percentage point "
        "(baseline CNN: 84.58% and 83.17%; MobileNetV2: 83.81% and 84.39% over two runs). "
        "Small differences between the first two models should therefore not be over-interpreted."
    )

    st.header("Image vs. environmental data (same 1,139 held-out test images)")
    st.table(pd.DataFrame(FUSION_ROWS, columns=["Model", "Test accuracy", "Note"]).set_index("Model"))

    st.header("Additional analysis")
    st.write(CV_TEXT)
    st.write(AMBIGUOUS_TEXT)
    st.write(FEATURE_TEXT)

    st.header("Limitations")
    st.markdown(
        "- Images are Hawaii-only (2014-2019); other regions are untested.\n"
        "- Temperature data is site-level (5 km grid, 12-week summary) while bleaching is labelled per colony, "
        "which caps what temperature can explain.\n"
        "- Only 7,591 of 10,419 images have usable environmental data.\n"
        "- The system classifies **current** bleaching status with environmental context. It does not forecast "
        "future bleaching; that would need repeat visits to the same colonies.\n"
        "- The fusion vs image-only difference is not statistically significant."
    )

# ---------------------------------------------------------------- Live demo
with tab_demo:
    st.header("Prediction demo")
    missing = [p for p in (MODEL_PATH, FEATURES_PATH, MANIFEST_PATH) if not os.path.exists(p)]
    if missing:
        st.error("Missing files: " + ", ".join(missing) + ". Re-run scripts 06 and 11 first.")
    else:
        full_model, extractor = load_keras_models()
        fusion = build_fusion()
        manifest = load_manifest()
        st.caption(
            f"The fusion classifier scores {fusion['test_acc'] * 100:.2f}% on "
            f"{fusion['n_test']:,} held-out test images it never trained on."
        )

        source = st.radio("Image source", ["Random held-out test image", "Upload my own photo"], horizontal=True)
        img, env, truth, info = None, None, None, ""

        if source == "Random held-out test image":
            test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
            if "sample_idx" not in st.session_state:
                st.session_state.sample_idx = int(np.random.randint(len(test_rows)))
            if st.button("Load another random image"):
                st.session_state.sample_idx = int(np.random.randint(len(test_rows)))
            row = test_rows.iloc[st.session_state.sample_idx]
            if os.path.exists(str(row["saved_path"])):
                img = Image.open(row["saved_path"])
                env = tuple(float(row[c]) for c in ENV_COLS)
                truth = str(row["label"])
                site = row["site_id"] if "site_id" in row.index else "unknown site"
                info = f"Held-out test image, site {site}"
            else:
                st.error(f"Image file not found: {row['saved_path']}")
        else:
            up = st.file_uploader("Coral photo (a close-up like the training images)", type=["jpg", "jpeg", "png"])
            if up is not None:
                img = Image.open(up)
                info = "Uploaded image"
            st.write("Sea-temperature data (enter the values for the reef site, averaged over the 12 weeks before the photo)")
            c1, c2, c3 = st.columns(3)
            dhw = c1.number_input("Max Degree Heating Weeks, last 12 wk (C-weeks)", 0.0, 30.0, 4.0, 0.5)
            sst = c2.number_input("Mean sea surface temperature, last 12 wk (C)", 15.0, 35.0, 27.5, 0.1)
            ssta = c3.number_input("Mean SST anomaly, last 12 wk (C)", -3.0, 5.0, 0.9, 0.1)
            env = (dhw, sst, ssta)

        if img is not None and env is not None:
            left, right = st.columns([1, 2])
            with left:
                st.image(img, caption=info, width=300)
            with right:
                p_img, p_fus = predict_all(img, env, full_model, extractor, fusion)
                m1, m2 = st.columns(2)
                for col, name, p in (
                    (m1, "Image only (EfficientNetB0)", p_img),
                    (m2, "Image + sea temperature (fusion)", p_fus),
                ):
                    col.markdown(f"**{name}**")
                    col.metric("Probability bleached", f"{p:.1%}")
                    col.progress(min(max(p, 0.0), 1.0))
                    col.write("Verdict: **BLEACHED**" if p >= 0.5 else "Verdict: **HEALTHY**")
                st.caption(
                    f"Sea-temperature inputs: max DHW {env[0]:.2f} C-weeks, mean SST {env[1]:.2f} C, "
                    f"mean SST anomaly {env[2]:.2f} C"
                )
                if truth:
                    st.info(f"Label assigned by NOAA scientists: **{truth.upper()}**")
            st.warning(
                "Research prototype trained on Hawaiian reef close-ups. Not for conservation decisions. "
                "Our evaluation found fusion is not significantly better than the image model alone."
            )
        elif img is None and source == "Upload my own photo":
            st.info("Upload a photo to begin.")
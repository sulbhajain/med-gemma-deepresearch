"""
Dataset loading, clinical scenario construction, and demographic augmentation.

Supported sources (tried in order):
  1. Kaggle input directory  — /kaggle/input/ultrasound-fetal-planes* CSV + PNG
  2. HuggingFace             — Idan0405/Fetal_Planes_DB
  3. MedMNIST BreastMNIST    — fallback when neither is available
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset

from config import (
    FETAL_PLANE_LABELS, PLANE_RISK_MAP, IMAGE_DIR,
    MAX_IMAGES, SEED,
)

random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_fetal_planes_kaggle() -> Optional[pd.DataFrame]:
    """Load Fetal Planes DB from a pre-attached Kaggle dataset."""
    kaggle_base = Path("/kaggle/input")
    candidates = [
        kaggle_base / "ultrasound-fetal-planes",
        kaggle_base / "fetal-health-classification",
        kaggle_base / "ultrasound-fetal-planes-dataset",
    ]
    for base in candidates:
        csvs = list(base.glob("**/*.csv"))
        imgs = list(base.glob("**/*.png")) + list(base.glob("**/*.jpg"))
        if csvs and imgs:
            print(f"   📁 Found Fetal Planes DB at {base}")
            df = pd.read_csv(csvs[0], sep=";")
            df.columns = df.columns.str.strip()
            img_dir = imgs[0].parent
            df["image_path"] = df["Image_name"].apply(
                lambda n: str(img_dir / f"{n}.png")
            )
            return df
    return None


def load_fetal_planes_huggingface(max_images: int = MAX_IMAGES) -> pd.DataFrame:
    """Download Fetal Planes DB via HuggingFace datasets hub."""
    print("   🌐 Fetching Fetal Planes DB from HuggingFace…")
    img_dir = Path(IMAGE_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Idan0405/Fetal_Planes_DB", split="train")
    records = []
    for i, sample in enumerate(tqdm(ds, desc="Saving images", total=min(len(ds), max_images))):
        if i >= max_images:
            break
        img   = sample["image"]
        label = sample["label"]
        plane = list(FETAL_PLANE_LABELS.keys())[label]
        out   = img_dir / f"fetal_{i:05d}.png"
        img.convert("RGB").save(out)
        records.append({
            "image_path":  str(out),
            "Plane":       plane,
            "label":       label,
            "US_Machine":  "Unknown",
            "Operator":    "Unknown",
        })
    return pd.DataFrame(records)


def load_medmnist_fallback(n: int = MAX_IMAGES) -> pd.DataFrame:
    """Fallback: MedMNIST BreastMNIST (ultrasound breast images)."""
    import medmnist
    from medmnist import BreastMNIST

    img_dir = Path(IMAGE_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)

    ds      = BreastMNIST(split="train", download=True, size=64)
    records = []
    for i, (img_arr, label_arr) in enumerate(ds):
        if i >= n:
            break
        label = int(label_arr[0])
        plane = "Fetal brain" if label == 1 else "Other"
        out   = img_dir / f"breast_{i:05d}.png"
        Image.fromarray(img_arr.squeeze()).convert("RGB").save(out)
        records.append({
            "image_path": str(out),
            "Plane":      plane,
            "label":      label,
        })
    return pd.DataFrame(records)


def load_dataset_auto() -> pd.DataFrame:
    """Try Kaggle → HuggingFace → MedMNIST in order."""
    df = load_fetal_planes_kaggle()
    if df is None:
        try:
            df = load_fetal_planes_huggingface()
        except Exception as e:
            print(f"   ⚠️  HuggingFace failed ({e}), using MedMNIST fallback")
            df = load_medmnist_fallback()
    df["risk_label"] = df["Plane"].map(PLANE_RISK_MAP).fillna("LOW")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Demographic augmentation
# ─────────────────────────────────────────────────────────────────────────────

DEMOGRAPHICS_POOL = [
    {"age": 22, "race_ethnicity": "Hispanic/Latina",       "insurance": "Medicaid",  "rural": True},
    {"age": 29, "race_ethnicity": "Non-Hispanic White",    "insurance": "Private",   "rural": False},
    {"age": 34, "race_ethnicity": "Non-Hispanic Black",    "insurance": "Medicaid",  "rural": False},
    {"age": 27, "race_ethnicity": "Asian",                 "insurance": "Private",   "rural": False},
    {"age": 31, "race_ethnicity": "Native American",       "insurance": "IHS",       "rural": True},
    {"age": 19, "race_ethnicity": "Non-Hispanic White",    "insurance": "Uninsured", "rural": True},
    {"age": 38, "race_ethnicity": "Hispanic/Latina",       "insurance": "Private",   "rural": False},
    {"age": 25, "race_ethnicity": "Non-Hispanic Black",    "insurance": "Medicaid",  "rural": True},
    {"age": 33, "race_ethnicity": "Pacific Islander",      "insurance": "Medicaid",  "rural": False},
    {"age": 28, "race_ethnicity": "Middle Eastern",        "insurance": "Private",   "rural": False},
]

CLINICAL_SCENARIOS: Dict[str, List[Dict]] = {
    "Fetal brain": [
        {
            "clinical_notes": "Third trimester ultrasound. Evaluating for ventriculomegaly. "
                              "Lateral ventricle width measured. HC plotted on growth curve.",
            "history":   "G2P1, family history of neural tube defects. 20-week anomaly scan normal.",
            "symptoms":  "Referred after borderline ventricular measurement on routine scan.",
            "findings":  ["Fetal brain imaging", "Ventricular measurement", "HC biometry"],
        },
        {
            "clinical_notes": "Fetal brain scan. Assessing corpus callosum and posterior fossa.",
            "history":   "G1P0, elevated quad-screen AFP.",
            "symptoms":  "Routine anomaly follow-up.",
            "findings":  ["Posterior fossa", "Cisterna magna", "Cerebellar vermis"],
        },
    ],
    "Fetal abdomen": [
        {
            "clinical_notes": "AC measured. Liver and stomach visualised. Umbilical vein insertion normal.",
            "history":   "G3P2, gestational diabetes on diet control.",
            "symptoms":  "Fundal height tracking above 90th centile.",
            "findings":  ["AC measurement", "Liver echogenicity", "Stomach bubble present"],
        },
    ],
    "Fetal femur": [
        {
            "clinical_notes": "Femur length measured for dating and growth. Long bone morphology assessed.",
            "history":   "G1P0, no skeletal dysplasia history.",
            "symptoms":  "Routine biometry at 28 weeks.",
            "findings":  ["FL biometry", "Long bone morphology normal"],
        },
    ],
    "Fetal thorax": [
        {
            "clinical_notes": "Four-chamber cardiac view obtained. Lung echogenicity vs liver compared. "
                              "Diaphragm integrity assessed.",
            "history":   "G2P1, maternal CMV seroconversion in first trimester.",
            "symptoms":  "Four-chamber view technically difficult on previous scan.",
            "findings":  ["Cardiac four-chamber view", "Lung-to-liver ratio", "Diaphragm"],
        },
    ],
    "Maternal cervix": [
        {
            "clinical_notes": "Transvaginal cervical length measurement. Funnelling assessed.",
            "history":   "G3P2L1, spontaneous preterm delivery at 32 weeks previously.",
            "symptoms":  "Pelvic pressure at 22 weeks.",
            "findings":  ["Cervical length", "Funnelling", "Internal os"],
        },
    ],
    "Other": [
        {
            "clinical_notes": "Unclassified or supplementary ultrasound view.",
            "history":   "Routine antenatal care.",
            "symptoms":  "No acute concerns.",
            "findings":  ["General survey"],
        },
    ],
}


def build_clinical_records(df: pd.DataFrame) -> List[Dict]:
    """Attach clinical scenarios and synthetic demographics to each image row."""
    records = []
    for i, row in df.iterrows():
        plane    = row.get("Plane", "Other")
        risk     = PLANE_RISK_MAP.get(plane, "LOW")
        scenario = random.choice(CLINICAL_SCENARIOS.get(plane, CLINICAL_SCENARIOS["Other"]))
        demo     = DEMOGRAPHICS_POOL[i % len(DEMOGRAPHICS_POOL)]

        records.append({
            "case_id":        f"CASE_{i:05d}",
            "image_path":     row["image_path"],
            "plane":          plane,
            "risk_level":     risk,
            "clinical_notes": scenario["clinical_notes"],
            "patient_history":scenario["history"],
            "symptoms":       scenario["symptoms"],
            "findings":       scenario["findings"],
            "demographics":   demo,
        })
    return records


def train_eval_test_split(records: List[Dict],
                          train_r=0.70, eval_r=0.15) -> tuple:
    random.shuffle(records)
    n       = len(records)
    n_train = int(n * train_r)
    n_eval  = int(n * eval_r)
    return (
        records[:n_train],
        records[n_train:n_train + n_eval],
        records[n_train + n_eval:],
    )

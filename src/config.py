"""
Central configuration for MedGemma Deep Research Agent.
Edit here to change model, dataset limits, and inference settings.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID       = "google/medgemma-4b-it"   # Full multimodal variant
USE_MULTIMODAL = True                       # False → text-only fallback

# ── Dataset ───────────────────────────────────────────────────────────────────
MAX_IMAGES        = 500   # Cap for quick Kaggle runs (full DB = 12,400)
TRAIN_RATIO       = 0.70
EVAL_RATIO        = 0.15
TEST_RATIO        = 0.15
SEED              = 42

# ── Image preprocessing ───────────────────────────────────────────────────────
TARGET_SIZE       = (448, 448)   # MedGemma vision encoder input
ENHANCE_CONTRAST  = True
REDUCE_NOISE      = True
CONTRAST_FACTOR   = 2.0
SHARPNESS_FACTOR  = 1.5
GAUSSIAN_RADIUS   = 0.8

# ── Inference ─────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS    = 512
TEMPERATURE       = 0.3
TOP_P             = 0.9

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR          = "./data"
IMAGE_DIR         = "./data/images"
RESULTS_DIR       = "./results"
MODELS_DIR        = "./models"

# ── Label maps ────────────────────────────────────────────────────────────────
FETAL_PLANE_LABELS = {
    "Fetal abdomen":   0,
    "Fetal brain":     1,
    "Fetal femur":     2,
    "Fetal thorax":    3,
    "Maternal cervix": 4,
    "Other":           5,
}

# Clinical risk derived from ultrasound plane type
PLANE_RISK_MAP = {
    "Fetal brain":     "HIGH",      # Ventriculomegaly, malformations
    "Maternal cervix": "HIGH",      # Cervical length → preterm risk
    "Fetal abdomen":   "MODERATE",  # AC, fetal growth
    "Fetal thorax":    "MODERATE",  # Cardiac, lung measurements
    "Fetal femur":     "LOW",       # Routine biometry
    "Other":           "LOW",
}

RISK_SCORE_MAP = {"HIGH": 0.90, "MODERATE": 0.55, "LOW": 0.20}
CONF_MAP       = {"high": 0.90, "medium": 0.60, "low": 0.30}

# ── Equity thresholds (used in equity report) ─────────────────────────────────
EQUITY_GEO_THRESHOLD  = 0.05   # Rural-urban gap ≤ 5% → equitable
EQUITY_RACE_THRESHOLD = 0.10   # Race gap ≤ 10% → equitable

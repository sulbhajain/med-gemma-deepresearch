# MedGemma Deep Research Agent
### Multimodal Maternal-Fetal Risk Triage with Equity Analysis

> **MedGemma Impact Challenge** submission · Kaggle 2025–2026  
> A four-phase multimodal AI agent that feeds real fetal ultrasound images into `google/medgemma-4b-it`'s vision encoder, cross-references clinical notes and EHR data, validates for health equity across race, geography, and insurance type, and outputs structured HIGH / MODERATE / LOW risk assessments.

---

## What makes this different

Most medical AI prototypes either (a) skip the image entirely and classify from text, or (b) run a black-box vision model without clinical context. This agent does both simultaneously — the vision encoder reads actual ultrasound pixels, the clinical correlation phase cross-references the imaging interpretation with patient history and symptoms, and a dedicated equity validation phase flags when social determinants of health (SDOH) may be affecting the recommendation.

---

## Architecture

```
Real Ultrasound Image (PNG)
         │
         ▼
┌─────────────────────────┐
│  Phase 1                │  MedGemma 4B-IT vision encoder
│  Visual Perception      │  → Identify plane, flag abnormalities
└─────────────┬───────────┘
              │ visual_findings
              ▼
┌─────────────────────────┐
│  Phase 2                │  MedGemma text reasoning
│  Clinical Correlation   │  + EHR: notes, history, symptoms
└─────────────┬───────────┘
              │ clinical_correlations
              ▼
┌─────────────────────────┐
│  Phase 3                │  Demographics: race, geography, insurance
│  Equity Validation      │  → SDOH-aware recommendations
└─────────────┬───────────┘
              │ equity_notes
              ▼
┌─────────────────────────┐
│  Phase 4                │  Structured output
│  Risk Stratification    │  → HIGH / MODERATE / LOW + confidence
└─────────────────────────┘
```

---

## Repository Structure

```
med-gemma-deepresearch/
├── notebooks/
│   └── med-gemma-deepresearch-kaggle.ipynb  # Self-contained Kaggle notebook
├── src/
│   ├── config.py          # All hyperparameters and label maps
│   ├── data.py            # Dataset loaders (Kaggle → HF → MedMNIST fallback)
│   ├── preprocessing.py   # CLAHE + despeckle pipeline for ultrasound images
│   ├── agent.py           # 4-phase DeepResearchAgent + ClinicalAssessment dataclass
│   ├── evaluate.py        # Batch evaluation + equity report + submission export
│   ├── visualise.py       # EDA, sample grid, dashboard, missed-risk panel
│   └── main.py            # End-to-end pipeline entry point
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1 — Install

```bash
git clone https://github.com/<your-username>/med-gemma-deepresearch.git
cd med-gemma-deepresearch
```

Preferred (`uv`):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

On Apple Silicon Macs, prefer Python 3.12 for best wheel compatibility with `torch`/`bitsandbytes`.

Alternative (`pip`):

```bash
pip install -r requirements.txt
```

### 2 — Authenticate (MedGemma requires gated HuggingFace access)

```bash
hf auth login
```

### 3 — Run the full pipeline

```bash
python src/main.py
# Outputs: results/submission.csv + 4 PNG dashboards
```

### 4 — Run a single case interactively

```python
from src.agent import DeepResearchAgent
from src.preprocessing import FetalUltrasoundPreprocessor
# ... load model and processor ...
agent = DeepResearchAgent(model, processor, device, FetalUltrasoundPreprocessor())
result = agent.assess(
    image_path      = "path/to/scan.png",
    clinical_notes  = "Third trimester, evaluating for ventriculomegaly.",
    patient_history = "G2P1, family history of NTDs.",
    symptoms        = "Borderline lateral ventricle measurement.",
    demographics    = {"age": 28, "race_ethnicity": "Hispanic/Latina",
                       "insurance": "Medicaid", "rural": True},
)
print(result.risk_level, result.reasoning)
```

### 5 — Launch Gradio app

```bash
python src/gradio_app.py
```

Then open the local URL printed in the terminal (usually `http://127.0.0.1:7860`).

---

## Dataset

Primary: **Fetal Planes DB** (Zenodo / Kaggle / HuggingFace `Idan0405/Fetal_Planes_DB`)
- 12,400 annotated fetal ultrasound images
- 6 planes: brain, abdomen, femur, thorax, maternal cervix, other
- Risk labels derived from clinical plane type

Fallback: **MedMNIST BreastMNIST** — real ultrasound images, automatically used if the Fetal Planes DB is unavailable.

---

## Image Preprocessing Pipeline

```
Raw PNG → Grayscale → Gaussian despeckle (σ=0.8)
       → PIL CLAHE approximation (contrast ×2.0, sharpness ×1.5)
       → RGB → Resize 448×448 → MedGemma vision encoder
```

---

## Equity Metrics

The agent tracks and reports three equity dimensions:

| Dimension | Threshold for "equitable" | Tracked variable |
|-----------|--------------------------|-----------------|
| Geographic | Rural-urban gap ≤ 5% | `rural` (bool) |
| Racial/ethnic | Max group gap ≤ 10% | `race_ethnicity` |
| Insurance | Reported, no hard threshold | `insurance` |

---

## Competition Alignment

| Criterion | How addressed |
|-----------|--------------|
| Effective use of HAI-DEF models | MedGemma 4B-IT multimodal — both vision encoder and language model |
| Problem importance | Maternal-fetal mortality disproportionately affects under-resourced populations |
| Real-world impact | Local deployment, no cloud dependency, equity-aware output |
| Technical feasibility | Runs on Kaggle free P100 in ~30 min |
| Execution & communication | Modular codebase, 4-panel dashboards, equity report |

---

## License

Apache 2.0. MedGemma weights are subject to Google's Health AI Developer Foundations Terms of Use.

> **Clinical disclaimer**: This is a research prototype. It must not be used to guide clinical decisions without prospective validation by qualified clinicians.

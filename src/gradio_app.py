"""
Gradio app for interactive maternal-fetal risk triage with MedGemma.

Usage
─────
    python src/gradio_app.py
"""

import os
import random
import re
import tempfile
from typing import Dict, Tuple

import gradio as gr
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    # BitsAndBytesConfig,
)

from agent import DeepResearchAgent
from config import MODEL_ID, USE_MULTIMODAL
from data import load_dataset_auto, build_clinical_records
from preprocessing import FetalUltrasoundPreprocessor


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
EXAMPLE_SEED = 42
EXAMPLE_RNG = random.Random(EXAMPLE_SEED)


PLACEHOLDER_VALUES = {
    "<identified plane>",
    "<2-3 sentences>",
    "<2–3 sentences>",
    "<1-2 sentences>",
    "<1–2 sentences>",
    "unknown",
}


def _normalize_text(value: str) -> str:
    text = (value or "").strip().strip("`\"")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_labeled_sections(value: str) -> str:
    lines = (value or "").splitlines()
    kept = []
    for line in lines:
        if re.match(r"^\s*(RISK|PLANE|CONFIDENCE|REASONING|RECOMMENDATION)\s*:", line, flags=re.IGNORECASE):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _clean_output_field(value: str, *, remove_sections: bool = False) -> str:
    text = _strip_labeled_sections(value) if remove_sections else (value or "")
    text = _normalize_text(text)
    if not text:
        return ""
    if text.lower() in PLACEHOLDER_VALUES:
        return ""
    if re.fullmatch(r"<[^>]+>", text):
        return ""
    return text


def load_model():
    # bnb = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    # )

    multimodal = USE_MULTIMODAL
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            # quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("✅ Multimodal MedGemma loaded")
    except Exception as e:
        print(f"⚠️ Multimodal load failed: {e} — falling back to text-only")
        multimodal = False
        processor = AutoTokenizer.from_pretrained(MODEL_ID)
        processor.pad_token = processor.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            # quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("✅ Text-only MedGemma loaded")

    device = next(model.parameters()).device
    print(f"Device: {device}")
    return model, processor, device, multimodal


print("Loading model and agent for Gradio app...")
MODEL, PROCESSOR, DEVICE, MULTIMODAL = load_model()
PREPROCESSOR = FetalUltrasoundPreprocessor()
AGENT = DeepResearchAgent(MODEL, PROCESSOR, DEVICE, PREPROCESSOR, MULTIMODAL)
print("✅ Gradio agent ready")


def _save_uploaded_image(image: Image.Image) -> str:
    if image is None:
        return ""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.convert("RGB").save(tmp.name)
        return tmp.name


def assess_case(
    image,
    clinical_notes: str,
    patient_history: str,
    symptoms: str,
    age: float,
    race_ethnicity: str,
    insurance: str,
    rural: bool,
    fast_mode: bool,
) -> Tuple[str, float, float, str, str, str, str, str, str, str]:
    if not clinical_notes.strip():
        return (
            "Please provide clinical notes.",
            0.0,
            0.0,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )

    image_path = _save_uploaded_image(image) if image is not None else ""

    demographics: Dict = {
        "age": int(age),
        "race_ethnicity": race_ethnicity.strip() or "unknown",
        "insurance": insurance.strip() or "unknown",
        "rural": bool(rural),
    }

    result = AGENT.assess(
        image_path=image_path,
        clinical_notes=clinical_notes,
        patient_history=patient_history,
        symptoms=symptoms,
        demographics=demographics,
        fast_mode=bool(fast_mode),
    )

    clean_plane = _clean_output_field(result.plane_identified)
    clean_reasoning = _clean_output_field(result.reasoning, remove_sections=True)
    clean_recommendation = _clean_output_field(result.recommendation, remove_sections=True)
    clean_equity = _clean_output_field(result.equity_notes, remove_sections=True)
    clean_differentials = _clean_output_field(result.differential_diagnoses)
    clean_uncertainty = _clean_output_field(result.uncertainty_summary)
    clean_safety = _clean_output_field(result.safety_flags)

    risk_label = (result.risk_level or "").upper()
    if result.cannot_assess:
        risk_label = f"{risk_label} (CANNOT ASSESS)"
    if result.review_required:
        risk_label = f"{risk_label} • REVIEW REQUIRED"

    return (
        risk_label,
        result.risk_score,
        result.confidence_score,
        clean_plane,
        clean_reasoning,
        clean_recommendation,
        clean_equity,
        clean_differentials,
        clean_uncertainty,
        clean_safety,
    )


STATIC_EXAMPLE = (
    None,
    "Third trimester ultrasound. Evaluating for ventriculomegaly with borderline ventricular measurement.",
    "G2P1, family history of neural tube defects. 20-week anomaly scan normal.",
    "Referred after borderline lateral ventricle measurement on routine scan.",
    28,
    "Hispanic/Latina",
    "Medicaid",
    True,
)


def _load_dataset_example_case():
    try:
        df = load_dataset_auto()
        records = build_clinical_records(df)
        if not records:
            return STATIC_EXAMPLE

        preferred = next((r for r in records if r.get("risk_level") == "HIGH"), records[0])
        demo = preferred.get("demographics", {})
        image = Image.open(preferred["image_path"]).convert("RGB")

        return (
            image,
            preferred.get("clinical_notes", ""),
            preferred.get("patient_history", ""),
            preferred.get("symptoms", ""),
            int(demo.get("age", 28)),
            str(demo.get("race_ethnicity", "Hispanic/Latina")),
            str(demo.get("insurance", "Medicaid")),
            bool(demo.get("rural", True)),
        )
    except Exception as e:
        print(f"⚠️ Could not load dataset example: {e}")
        return STATIC_EXAMPLE


EXAMPLE_CASE = _load_dataset_example_case()


def load_example_case(example_tier: str):
    try:
        df = load_dataset_auto()
        records = build_clinical_records(df)
        if not records:
            return EXAMPLE_CASE

        tier = (example_tier or "AUTO").upper()
        if tier == "AUTO":
            candidate_pool = [r for r in records if r.get("risk_level") in {"HIGH", "MODERATE"}]
        elif tier == "ANY":
            candidate_pool = records
        else:
            candidate_pool = [r for r in records if r.get("risk_level") == tier]

        selected = EXAMPLE_RNG.choice(candidate_pool if candidate_pool else records)
        demo = selected.get("demographics", {})
        image = Image.open(selected["image_path"]).convert("RGB")

        return (
            image,
            selected.get("clinical_notes", ""),
            selected.get("patient_history", ""),
            selected.get("symptoms", ""),
            int(demo.get("age", 28)),
            str(demo.get("race_ethnicity", "Hispanic/Latina")),
            str(demo.get("insurance", "Medicaid")),
            bool(demo.get("rural", True)),
        )
    except Exception as e:
        print(f"⚠️ Could not rotate dataset example: {e}")
        return EXAMPLE_CASE


with gr.Blocks(title="MedGemma Deep Research") as APP:
    gr.Markdown("# MedGemma Deep Research Agent")
    gr.Markdown("Interactive maternal-fetal risk triage (HIGH / MODERATE / LOW)")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Ultrasound Image")
            clinical_notes_input = gr.Textbox(
                label="Clinical Notes",
                lines=4,
                placeholder="Enter ultrasound/clinical notes...",
            )
            history_input = gr.Textbox(label="Patient History", lines=3)
            symptoms_input = gr.Textbox(label="Symptoms", lines=3)

            age_input = gr.Slider(minimum=14, maximum=50, value=28, step=1, label="Age")
            race_input = gr.Textbox(label="Race / Ethnicity", value="Hispanic/Latina")
            insurance_input = gr.Textbox(label="Insurance", value="Medicaid")
            rural_input = gr.Checkbox(label="Rural residence", value=True)
            fast_mode_input = gr.Checkbox(label="Fast Mode (lower latency)", value=True)

            with gr.Row():
                example_button = gr.Button("Load Example")
                run_button = gr.Button("Assess Risk", variant="primary")

            example_tier_input = gr.Dropdown(
                label="Example Risk Tier",
                choices=["AUTO", "HIGH", "MODERATE", "LOW", "ANY"],
                value="AUTO",
            )

        with gr.Column():
            risk_level_out = gr.Textbox(label="Risk Level")
            risk_score_out = gr.Number(label="Risk Score")
            conf_score_out = gr.Number(label="Confidence Score")
            plane_out = gr.Textbox(label="Plane Identified")
            reasoning_out = gr.Textbox(label="Reasoning", lines=6)
            recommendation_out = gr.Textbox(label="Recommendation", lines=4)
            equity_out = gr.Textbox(label="Equity Notes", lines=4)
            differential_out = gr.Textbox(label="Differential Diagnoses", lines=5)
            uncertainty_out = gr.Textbox(label="Uncertainty Decomposition", lines=3)
            safety_out = gr.Textbox(label="Safety Flags", lines=2)

    run_button.click(
        fn=assess_case,
        inputs=[
            image_input,
            clinical_notes_input,
            history_input,
            symptoms_input,
            age_input,
            race_input,
            insurance_input,
            rural_input,
            fast_mode_input,
        ],
        outputs=[
            risk_level_out,
            risk_score_out,
            conf_score_out,
            plane_out,
            reasoning_out,
            recommendation_out,
            equity_out,
            differential_out,
            uncertainty_out,
            safety_out,
        ],
    )

    example_button.click(
        fn=load_example_case,
        inputs=[example_tier_input],
        outputs=[
            image_input,
            clinical_notes_input,
            history_input,
            symptoms_input,
            age_input,
            race_input,
            insurance_input,
            rural_input,
        ],
    )


if __name__ == "__main__":
    APP.launch()

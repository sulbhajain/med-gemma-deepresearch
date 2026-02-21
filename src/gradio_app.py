"""
Gradio app for interactive maternal-fetal risk triage with MedGemma.

Usage
─────
    python src/gradio_app.py
"""

import os
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
from preprocessing import FetalUltrasoundPreprocessor


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


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
) -> Tuple[str, float, float, str, str, str, str]:
    if not clinical_notes.strip():
        return (
            "Please provide clinical notes.",
            0.0,
            0.0,
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
    )

    return (
        result.risk_level,
        result.risk_score,
        result.confidence_score,
        result.plane_identified,
        result.reasoning,
        result.recommendation,
        result.equity_notes,
    )


def load_example_case():
    return (
        None,
        "Third trimester ultrasound. Evaluating for ventriculomegaly with borderline ventricular measurement.",
        "G2P1, family history of neural tube defects. 20-week anomaly scan normal.",
        "Referred after borderline lateral ventricle measurement on routine scan.",
        28,
        "Hispanic/Latina",
        "Medicaid",
        True,
    )


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

            with gr.Row():
                example_button = gr.Button("Load Example")
                run_button = gr.Button("Assess Risk", variant="primary")

        with gr.Column():
            risk_level_out = gr.Textbox(label="Risk Level")
            risk_score_out = gr.Number(label="Risk Score")
            conf_score_out = gr.Number(label="Confidence Score")
            plane_out = gr.Textbox(label="Plane Identified")
            reasoning_out = gr.Textbox(label="Reasoning", lines=6)
            recommendation_out = gr.Textbox(label="Recommendation", lines=4)
            equity_out = gr.Textbox(label="Equity Notes", lines=4)

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
        ],
        outputs=[
            risk_level_out,
            risk_score_out,
            conf_score_out,
            plane_out,
            reasoning_out,
            recommendation_out,
            equity_out,
        ],
    )

    example_button.click(
        fn=load_example_case,
        inputs=[],
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

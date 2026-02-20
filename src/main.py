"""
MedGemma Deep Research Agent — End-to-End Pipeline

Runs the full pipeline:
  1. Load Fetal Planes DB (Kaggle → HuggingFace → MedMNIST fallback)
  2. Build clinical records with demographics
  3. Load MedGemma 4B-IT (multimodal)
  4. Baseline demo on 3 representative cases
  5. Batch evaluation (20 test cases)
  6. Equity analysis report
  7. Save submission.csv and visualisation PNGs

Usage
─────
    python src/main.py
"""

import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODEL_ID, USE_MULTIMODAL, RESULTS_DIR
from data import load_dataset_auto, build_clinical_records, train_eval_test_split
from preprocessing import FetalUltrasoundPreprocessor
from agent import DeepResearchAgent
from evaluate import run_batch_evaluation, classification_summary, equity_report, save_submission
from visualise import eda_overview, sample_grid, evaluation_dashboard, missed_high_risk


def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    multimodal = USE_MULTIMODAL
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("✅ Multimodal MedGemma loaded (image + text)")
    except Exception as e:
        print(f"   ⚠️  Multimodal failed: {e} — falling back to text-only")
        multimodal = False
        processor = AutoTokenizer.from_pretrained(MODEL_ID)
        processor.pad_token = processor.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("✅ Text-only MedGemma loaded")

    device = next(model.parameters()).device
    print(f"   Device: {device}  |  Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    return model, processor, device, multimodal


def main():
    # ── 1. Dataset ──────────────────────────────────────────────────────────
    print("\n⚙️  Loading dataset…")
    df_raw = load_dataset_auto()
    print(f"   Loaded {len(df_raw)} images")
    eda_overview(df_raw)

    print("\n⚙️  Building clinical records…")
    all_records = build_clinical_records(df_raw)
    train_records, eval_records, test_records = train_eval_test_split(all_records)
    print(f"   Train={len(train_records)}  Eval={len(eval_records)}  Test={len(test_records)}")

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    preprocessor = FetalUltrasoundPreprocessor()
    sample_grid(all_records, preprocessor, n=12)

    # ── 3. Model ─────────────────────────────────────────────────────────────
    print("\n⚙️  Loading MedGemma…")
    model, processor, device, multimodal = load_model()

    agent = DeepResearchAgent(model, processor, device, preprocessor, multimodal)
    print(f"✅ Agent ready  (mode: {'multimodal' if multimodal else 'text-only'})")

    # ── 4. Demo ───────────────────────────────────────────────────────────────
    print("\n🔬 Running demo on 3 representative cases…")
    seen = {}
    for rec in all_records:
        if rec["risk_level"] not in seen:
            seen[rec["risk_level"]] = rec
        if len(seen) == 3:
            break

    for risk_label, case in seen.items():
        print(f"\n{'='*70}\nDEMO — Expected: {risk_label}")
        result = agent.assess(
            case["image_path"], case["clinical_notes"],
            case["patient_history"], case["symptoms"], case["demographics"],
        )
        match = "✅ CORRECT" if result.risk_level == risk_label else "❌ INCORRECT"
        print(f"  Predicted: {result.risk_level} ({result.risk_score:.2f}) — {match}")
        print(f"  Reasoning: {result.reasoning[:200]}")

    # ── 5. Batch evaluation ──────────────────────────────────────────────────
    print("\n📊 Running batch evaluation…")
    results_df = run_batch_evaluation(agent, test_records, max_cases=20)
    classification_summary(results_df)

    # ── 6. Equity report ─────────────────────────────────────────────────────
    metrics = equity_report(results_df)

    # ── 7. Visualise & save ──────────────────────────────────────────────────
    evaluation_dashboard(
        results_df,
        results_df["actual_risk"].tolist(),
        results_df["predicted_risk"].tolist(),
    )
    missed_high_risk(results_df, preprocessor)
    save_submission(results_df, f"{RESULTS_DIR}/submission.csv")

    print(f"\n{'='*70}\nPIPELINE COMPLETE")
    print(f"  Overall accuracy       : {metrics['overall_accuracy']:.2%}")
    print(f"  Rural-urban equity gap : {metrics['geo_gap']:.2%}")
    print(f"  Racial equity gap      : {metrics['race_gap']:.2%}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from config import MODEL_ID, USE_MULTIMODAL
from data import load_dataset_auto, build_clinical_records, train_eval_test_split
from preprocessing import FetalUltrasoundPreprocessor
from agent import DeepResearchAgent
from evaluate import run_batch_evaluation


def main():
    print("Loading dataset...")
    df = load_dataset_auto()
    records = build_clinical_records(df)
    _, _, test_records = train_eval_test_split(records)
    records3 = test_records[:3]
    print(f"Smoke records: {len(records3)}")

    multimodal = USE_MULTIMODAL
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype="auto",
        )
        print("Loaded multimodal model")
    except Exception as exc:
        print(f"Multimodal load failed: {exc}; falling back to text-only")
        multimodal = False
        processor = AutoTokenizer.from_pretrained(MODEL_ID)
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype="auto",
        )
        print("Loaded text-only model")

    device = next(model.parameters()).device
    agent = DeepResearchAgent(model, processor, device, FetalUltrasoundPreprocessor(), multimodal)

    results = run_batch_evaluation(agent, records3, max_cases=3)
    out_path = Path("results/real_smoke_eval.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    expected_columns = [
        "differential_diagnoses",
        "uncertainty_summary",
        "safety_flags",
        "review_required",
        "cannot_assess",
    ]
    print("\nColumns present:")
    for col in expected_columns:
        print(f"  {col}: {col in results.columns}")

    print("\nSample output rows:")
    sample = results[[
        "case_id",
        "predicted_risk",
        "differential_diagnoses",
        "uncertainty_summary",
        "safety_flags",
        "review_required",
        "cannot_assess",
    ]]
    print(sample.to_string(index=False))

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from agent import DeepResearchAgent
from config import MODEL_ID, USE_MULTIMODAL
from data import build_clinical_records, load_dataset_auto, train_eval_test_split
from evaluate import run_batch_evaluation
from preprocessing import FetalUltrasoundPreprocessor

BATCH_CASES = 10
LATENCY_CASES = 3


def load_agent():
    multimodal = USE_MULTIMODAL
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype="auto",
        )
    except Exception:
        multimodal = False
        processor = AutoTokenizer.from_pretrained(MODEL_ID)
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype="auto",
        )

    device = next(model.parameters()).device
    return DeepResearchAgent(model, processor, device, FetalUltrasoundPreprocessor(), multimodal)


def compute_core_metrics(df: pd.DataFrame):
    overall_agreement = float(df["correct"].mean())

    high_df = df[df["actual_risk"] == "HIGH"]
    if len(high_df) == 0:
        high_risk_sensitivity = float("nan")
    else:
        high_risk_sensitivity = float((high_df["predicted_risk"] == "HIGH").mean())

    rural_acc = df[df["rural"] == True]["correct"].mean()
    urban_acc = df[df["rural"] == False]["correct"].mean()
    rural_urban_gap = float(abs(rural_acc - urban_acc))

    race_tbl = df.groupby("race_ethnicity")["correct"].mean()
    race_gap = float(race_tbl.max() - race_tbl.min()) if len(race_tbl) > 1 else 0.0

    return {
        "high_risk_sensitivity": high_risk_sensitivity,
        "overall_triage_agreement": overall_agreement,
        "rural_urban_gap": rural_urban_gap,
        "max_race_ethnicity_gap": race_gap,
    }


def compute_latency(agent: DeepResearchAgent, cases):
    full_times = []
    fast_times = []

    for case in cases:
        start = time.perf_counter()
        _ = agent.assess(
            image_path=case["image_path"],
            clinical_notes=case["clinical_notes"],
            patient_history=case["patient_history"],
            symptoms=case["symptoms"],
            demographics=case["demographics"],
            fast_mode=False,
        )
        full_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        _ = agent.assess(
            image_path=case["image_path"],
            clinical_notes=case["clinical_notes"],
            patient_history=case["patient_history"],
            symptoms=case["symptoms"],
            demographics=case["demographics"],
            fast_mode=True,
        )
        fast_times.append(time.perf_counter() - start)

    full_median = float(np.median(full_times))
    fast_median = float(np.median(fast_times))
    improvement = float(((full_median - fast_median) / full_median) * 100.0) if full_median > 0 else 0.0

    return {
        "full_mode_median_sec": full_median,
        "fast_mode_median_sec": fast_median,
        "improvement_percent": improvement,
        "full_mode_times_sec": full_times,
        "fast_mode_times_sec": fast_times,
    }


def main():
    print("Loading dataset and records...")
    df_raw = load_dataset_auto()
    records = build_clinical_records(df_raw)
    _, _, test_records = train_eval_test_split(records)

    eval_records = test_records[:BATCH_CASES]
    latency_records = test_records[:LATENCY_CASES]

    print(f"Evaluation cases: {len(eval_records)} | Latency cases: {len(latency_records)}")

    print("Loading model/agent...")
    agent = load_agent()

    print("Running batch evaluation...")
    results_df = run_batch_evaluation(agent, eval_records, max_cases=BATCH_CASES)

    print("Computing latency benchmarks...")
    latency_metrics = compute_latency(agent, latency_records)
    core_metrics = compute_core_metrics(results_df)

    output = {
        "sample_sizes": {
            "batch_cases": len(eval_records),
            "latency_cases": len(latency_records),
        },
        "metrics": {**core_metrics, **latency_metrics},
    }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "kaggle_metrics.json"
    csv_path = out_dir / "kaggle_metrics_eval_rows.csv"

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    results_df.to_csv(csv_path, index=False)

    print("\n=== METRICS ===")
    print(f"High-risk sensitivity: {core_metrics['high_risk_sensitivity']*100:.2f}%")
    print(f"Overall triage agreement: {core_metrics['overall_triage_agreement']*100:.2f}%")
    print(f"Rural-urban gap: {core_metrics['rural_urban_gap']*100:.2f}%")
    print(f"Max race/ethnicity gap: {core_metrics['max_race_ethnicity_gap']*100:.2f}%")
    print(f"Median interactive latency (Full): {latency_metrics['full_mode_median_sec']:.2f} sec")
    print(f"Median interactive latency (Fast): {latency_metrics['fast_mode_median_sec']:.2f} sec")
    print(f"Latency improvement: {latency_metrics['improvement_percent']:.2f}%")
    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved eval rows CSV: {csv_path}")


if __name__ == "__main__":
    main()

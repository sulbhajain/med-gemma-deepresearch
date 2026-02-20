"""
Batch evaluation and equity analysis for the Deep Research Agent.

Usage
─────
    from evaluate import run_batch_evaluation, equity_report, save_submission
    results_df = run_batch_evaluation(agent, test_records)
    equity_report(results_df)
    save_submission(results_df, "results/submission.csv")
"""

from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

from agent import DeepResearchAgent


ALL_LABELS = ["HIGH", "MODERATE", "LOW"]


def run_batch_evaluation(
    agent:       DeepResearchAgent,
    records:     List[Dict],
    max_cases:   int = None,
) -> pd.DataFrame:
    """
    Run agent.assess() on every record and return a results DataFrame.

    Columns: case_id, plane, predicted_risk, risk_score, confidence,
             reasoning, equity_notes, image_path, actual_risk, correct,
             + all demographics keys.
    """
    if max_cases:
        records = records[:max_cases]

    rows, ground_truths = [], []

    for case in tqdm(records, desc="Evaluating"):
        try:
            result = agent.assess(
                image_path      = case["image_path"],
                clinical_notes  = case["clinical_notes"],
                patient_history = case["patient_history"],
                symptoms        = case["symptoms"],
                demographics    = case["demographics"],
            )
            rows.append({
                "case_id":        case["case_id"],
                "plane":          case["plane"],
                "predicted_risk": result.risk_level,
                "risk_score":     result.risk_score,
                "confidence":     result.confidence_score,
                "reasoning":      result.reasoning,
                "equity_notes":   result.equity_notes,
                "image_path":     case["image_path"],
                **case["demographics"],
            })
        except Exception as e:
            print(f"   ⚠️  {case['case_id']}: {e}")
            rows.append({
                "case_id":        case["case_id"],
                "plane":          case["plane"],
                "predicted_risk": "MODERATE",
                "risk_score":     0.5,
                "confidence":     0.3,
                "reasoning":      "Error during inference",
                "equity_notes":   "N/A",
                "image_path":     case["image_path"],
                **case["demographics"],
            })
        ground_truths.append(case["risk_level"])

    df = pd.DataFrame(rows)
    df["actual_risk"] = ground_truths
    df["correct"]     = df["predicted_risk"] == df["actual_risk"]
    return df


def classification_summary(df: pd.DataFrame):
    """Print sklearn classification report."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION PERFORMANCE")
    print("=" * 70 + "\n")
    print(classification_report(
        df["actual_risk"], df["predicted_risk"],
        labels=ALL_LABELS, target_names=ALL_LABELS,
    ))


def equity_report(df: pd.DataFrame) -> Dict:
    """
    Print and return equity metrics across geography, race, and insurance.
    Returns a dict with gap values for downstream assertions.
    """
    overall  = df["correct"].mean()
    print(f"\n📊 Overall accuracy : {overall:.2%}")
    print(f"   Mean confidence : {df['confidence'].mean():.2f}")

    # Geographic equity
    rural_acc = df[df["rural"] == True]["correct"].mean()
    urban_acc = df[df["rural"] == False]["correct"].mean()
    geo_gap   = abs(rural_acc - urban_acc)
    print(f"\n📍 Geographic equity")
    print(f"   Rural  : {rural_acc:.2%}")
    print(f"   Urban  : {urban_acc:.2%}")
    print(f"   Gap    : {geo_gap:.2%}  "
          f"{'✅ Equitable (<5%)' if geo_gap < 0.05 else '⚠️  Bias detected (≥5%)'}")

    # Race / ethnicity
    race_tbl  = df.groupby("race_ethnicity")["correct"].mean().sort_values(ascending=False)
    race_gap  = race_tbl.max() - race_tbl.min()
    print(f"\n🌍 Racial/ethnic equity")
    for race, acc in race_tbl.items():
        bar = "█" * int(acc * 20)
        print(f"   {race:<30} {acc:.2%}  {bar}")
    print(f"   Max gap: {race_gap:.2%}  "
          f"{'✅ Equitable (<10%)' if race_gap < 0.10 else '⚠️  Disparate (≥10%)'}")

    # Insurance
    ins_tbl = df.groupby("insurance")["correct"].mean().sort_values(ascending=False)
    print(f"\n💳 Insurance equity")
    for ins, acc in ins_tbl.items():
        print(f"   {ins:<15} {acc:.2%}")

    return {
        "overall_accuracy": overall,
        "geo_gap":          geo_gap,
        "race_gap":         race_gap,
    }


def save_submission(df: pd.DataFrame, path: str = "results/submission.csv"):
    """Save competition submission CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sub = df[[
        "case_id", "predicted_risk", "risk_score",
        "confidence", "plane", "reasoning", "equity_notes",
    ]].rename(columns={"predicted_risk": "risk_level", "equity_notes": "equity_note"})
    sub["image_used"] = True
    sub["risk_score"]  = sub["risk_score"].round(3)
    sub["confidence"]  = sub["confidence"].round(3)
    sub["reasoning"]   = sub["reasoning"].str[:200]
    sub["equity_note"] = sub["equity_note"].str[:150]
    sub.to_csv(path, index=False)
    print(f"✅ Submission saved → {path}  ({len(sub)} rows)")
    return sub

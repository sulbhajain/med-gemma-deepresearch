"""
Visualisation utilities: EDA overview, sample grid, evaluation dashboard,
and missed-high-risk review panel.
"""

from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix

from preprocessing import FetalUltrasoundPreprocessor
from config import RESULTS_DIR

RISK_COLOR = {"HIGH": "#e74c3c", "MODERATE": "#f39c12", "LOW": "#2ecc71"}


def eda_overview(df: pd.DataFrame, save: bool = True):
    """Plane distribution + risk pie + sample image grid."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Fetal Planes DB — Exploratory Data Analysis",
                 fontsize=14, fontweight="bold")

    # Plane distribution
    if "Plane" in df.columns:
        pc = df["Plane"].value_counts()
        axes[0].barh(pc.index, pc.values, color=sns.color_palette("Set2"))
        axes[0].set_title("Image Plane Distribution")
        axes[0].set_xlabel("Count")
        for i, v in enumerate(pc.values):
            axes[0].text(v + 2, i, str(v), va="center", fontsize=9)

    # Risk pie
    if "risk_label" in df.columns:
        rc     = df["risk_label"].value_counts()
        colors = [RISK_COLOR.get(r, "grey") for r in rc.index]
        axes[1].pie(rc.values, labels=rc.index, colors=colors,
                    autopct="%1.1f%%", startangle=90)
        axes[1].set_title("Risk Level Distribution\n(derived from plane type)")

    # Sample images
    axes[2].axis("off")
    if "image_path" in df.columns:
        sample_paths = (
            df.dropna(subset=["image_path"])
              .groupby("Plane")
              .first()["image_path"]
              .tolist()[:6]
        )
        sub_gs = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=axes[2].get_subplotspec(), hspace=0.3, wspace=0.3
        )
        fig.delaxes(axes[2])
        for idx, path in enumerate(sample_paths[:6]):
            try:
                ax = fig.add_subplot(sub_gs[idx // 3, idx % 3])
                ax.imshow(Image.open(path).convert("L"), cmap="gray")
                ax.axis("off")
                row = df[df["image_path"] == path]
                ax.set_title(
                    row["Plane"].values[0] if len(row) else "",
                    fontsize=7, pad=2,
                )
            except Exception:
                pass

    plt.tight_layout()
    if save:
        out = Path(RESULTS_DIR) / "eda_overview.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✅ EDA saved → {out}")
    plt.show()


def sample_grid(records: List[Dict], preprocessor: FetalUltrasoundPreprocessor,
                n: int = 12, save: bool = True):
    """Grid of ultrasound thumbnails with risk/demographic labels."""
    cols = 4
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(18, 4 * rows_n))
    fig.suptitle("Real Fetal Ultrasound Images — Clinical Dataset Samples",
                 fontsize=14, fontweight="bold")

    for idx, (ax, case) in enumerate(zip(axes.flat, records[:n])):
        try:
            ax.imshow(preprocessor(case["image_path"]), cmap="gray")
        except Exception:
            ax.imshow(np.zeros((448, 448)), cmap="gray")
        risk = case["risk_level"]
        demo = case["demographics"]
        ax.set_title(
            f"{case['plane']}\n{risk} RISK | Age {demo['age']}\n"
            f"{demo['race_ethnicity']} | {'Rural' if demo['rural'] else 'Urban'}",
            fontsize=8, color=RISK_COLOR.get(risk, "black"), fontweight="bold",
        )
        ax.axis("off")

    # Hide unused axes
    for ax in axes.flat[len(records[:n]):]:
        ax.axis("off")

    plt.tight_layout()
    if save:
        out = Path(RESULTS_DIR) / "dataset_samples.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✅ Sample grid saved → {out}")
    plt.show()


def evaluation_dashboard(df: pd.DataFrame, ground_truths: List[str],
                          pred_risks: List[str], save: bool = True):
    """6-panel evaluation dashboard."""
    ALL_LABELS = ["HIGH", "MODERATE", "LOW"]

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm  = confusion_matrix(ground_truths, pred_risks, labels=ALL_LABELS)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=ALL_LABELS, yticklabels=ALL_LABELS)
    ax1.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual"); ax1.set_xlabel("Predicted")

    # 2. Accuracy by plane
    ax2 = fig.add_subplot(gs[0, 1])
    plane_acc = df.groupby("plane")["correct"].mean().sort_values()
    ax2.barh(plane_acc.index, plane_acc.values,
             color=[("#2ecc71" if v >= 0.7 else "#e74c3c") for v in plane_acc.values])
    ax2.set_xlim(0, 1)
    ax2.set_title("Accuracy by Ultrasound Plane", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Accuracy")
    for i, v in enumerate(plane_acc.values):
        ax2.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)

    # 3. Risk score distribution
    ax3 = fig.add_subplot(gs[0, 2])
    for risk, color in [("HIGH", "#e74c3c"), ("MODERATE", "#f39c12"), ("LOW", "#2ecc71")]:
        sub = df[df["predicted_risk"] == risk]["risk_score"]
        ax3.hist(sub, bins=15, alpha=0.6, color=color, label=risk)
    ax3.set_title("Risk Score Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Risk Score"); ax3.legend()

    # 4. Accuracy by race/ethnicity
    ax4 = fig.add_subplot(gs[1, 0])
    race_acc = df.groupby("race_ethnicity")["correct"].mean().sort_values()
    ax4.barh(race_acc.index, race_acc.values,
             color=sns.color_palette("Set2", len(race_acc)))
    ax4.set_xlim(0, 1)
    ax4.axvline(race_acc.mean(), color="red", linestyle="--", alpha=0.7, label="Mean")
    ax4.set_title("Accuracy by Race/Ethnicity\n(Equity Metric)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Accuracy"); ax4.legend(fontsize=8)

    # 5. Rural vs Urban
    ax5 = fig.add_subplot(gs[1, 1])
    loc_acc = df.groupby("rural")["correct"].mean()
    ax5.bar(["Urban", "Rural"],
            [loc_acc.get(False, 0), loc_acc.get(True, 0)],
            color=["#3498db", "#e67e22"])
    ax5.set_ylim(0, 1)
    ax5.set_title("Rural vs Urban Accuracy", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Accuracy")

    # 6. Insurance breakdown
    ax6 = fig.add_subplot(gs[1, 2])
    ins_acc = df.groupby("insurance")["correct"].mean().sort_values()
    ax6.barh(ins_acc.index, ins_acc.values,
             color=sns.color_palette("Paired", len(ins_acc)))
    ax6.set_xlim(0, 1)
    ax6.set_title("Accuracy by Insurance Type\n(Equity Metric)", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Accuracy")

    plt.suptitle("Deep Research Agent — Evaluation Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    if save:
        out = Path(RESULTS_DIR) / "evaluation_dashboard.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✅ Dashboard saved → {out}")
    plt.show()


def missed_high_risk(df: pd.DataFrame, preprocessor: FetalUltrasoundPreprocessor,
                     save: bool = True):
    """Display cases where HIGH risk was incorrectly predicted as lower."""
    missed = df[(df["actual_risk"] == "HIGH") & (df["predicted_risk"] != "HIGH")]
    print(f"Missed HIGH-risk cases: {len(missed)} / {(df['actual_risk']=='HIGH').sum()}")

    n_show = min(4, len(missed))
    if n_show == 0:
        print("🎉 No high-risk cases missed!")
        return

    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, missed.head(n_show).iterrows()):
        try:
            ax.imshow(preprocessor(row["image_path"]), cmap="gray")
        except Exception:
            ax.imshow(np.zeros((448, 448)), cmap="gray")
        ax.set_title(
            f"Plane: {row['plane']}\nActual: HIGH | Pred: {row['predicted_risk']}\n"
            f"Conf: {row['confidence']:.0%} | Age {row['age']}",
            fontsize=9, color="red",
        )
        ax.axis("off")

    plt.suptitle("Missed High-Risk Cases", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        out = Path(RESULTS_DIR) / "missed_high_risk.png"
        plt.savefig(out, dpi=150)
        print(f"✅ Saved → {out}")
    plt.show()

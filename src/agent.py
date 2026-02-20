"""
Multi-Phase Deep Research Agent for Maternal-Fetal Risk Triage.

Architecture
────────────
Phase 1  Visual Perception     MedGemma vision encoder reads real ultrasound image
Phase 2  Clinical Correlation  Cross-references imaging findings with EHR data
Phase 3  Equity Validation     Social-determinant-aware recommendation layer
Phase 4  Risk Stratification   Final structured HIGH / MODERATE / LOW output

Usage
─────
    agent = DeepResearchAgent(model, processor, device, preprocessor)
    result: ClinicalAssessment = agent.assess(
        image_path      = "path/to/scan.png",
        clinical_notes  = "...",
        patient_history = "...",
        symptoms        = "...",
        demographics    = {"age": 28, "race_ethnicity": "...", "rural": True, ...},
    )
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from preprocessing import FetalUltrasoundPreprocessor
from config import RISK_SCORE_MAP, CONF_MAP, MAX_NEW_TOKENS, TEMPERATURE, TOP_P


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClinicalAssessment:
    """Structured output returned by DeepResearchAgent.assess()."""
    risk_level:            str
    risk_score:            float
    plane_identified:      str
    visual_findings:       str
    clinical_correlations: str
    equity_notes:          str
    reasoning:             str
    recommendation:        str
    confidence_score:      float
    image_used:            bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class DeepResearchAgent:
    """Four-phase multimodal clinical reasoning agent."""

    def __init__(
        self,
        model,
        processor,
        device,
        preprocessor: FetalUltrasoundPreprocessor,
        multimodal: bool = True,
    ):
        self.model        = model
        self.processor    = processor
        self.device       = device
        self.preprocessor = preprocessor
        self.multimodal   = multimodal

        # Resolve tokenizer regardless of processor type
        self._tok = getattr(processor, "tokenizer", processor)
        if self._tok.pad_token_id is None:
            self._tok.pad_token_id = self._tok.eos_token_id

    # ── Public API ────────────────────────────────────────────────────────────

    def assess(
        self,
        image_path:      str,
        clinical_notes:  str,
        patient_history: str = "",
        symptoms:        str = "",
        demographics:    Optional[Dict] = None,
    ) -> ClinicalAssessment:
        demographics = demographics or {}
        visual  = self._visual_perception(image_path)
        corr    = self._clinical_correlation(visual, clinical_notes,
                                              patient_history, symptoms, demographics)
        equity  = self._equity_validation(corr, demographics)
        return self._risk_stratification(visual, corr, equity,
                                         clinical_notes, patient_history,
                                         symptoms, demographics)

    # ── Phase 1: Visual Perception ────────────────────────────────────────────

    def _visual_perception(self, image_path: str) -> str:
        img = self.preprocessor(image_path)

        prompt = (
            "You are a medical imaging specialist reviewing an ultrasound image.\n\n"
            "### Tasks\n"
            "1. Identify the anatomical plane (fetal brain / abdomen / femur / thorax / "
            "maternal cervix / other).\n"
            "2. Note key visible structures.\n"
            "3. Flag any obvious abnormalities.\n\n"
            "Respond in 3–5 sentences."
        )

        if self.multimodal:
            inputs = self.processor(
                text=[prompt], images=[img], return_tensors="pt", padding=True
            ).to(self.device)
        else:
            inputs = self._tok(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=TOP_P,
                pad_token_id=self._tok.pad_token_id,
            )
        return self._tok.decode(ids[0], skip_special_tokens=True)

    # ── Phase 2: Clinical Correlation ─────────────────────────────────────────

    def _clinical_correlation(
        self,
        visual:         str,
        clinical_notes: str,
        history:        str,
        symptoms:       str,
        demographics:   Dict,
    ) -> str:
        age   = demographics.get("age", "unknown")
        insur = demographics.get("insurance", "unknown")
        rural = "rural" if demographics.get("rural") else "urban"

        prompt = (
            f"Patient: {age}y, {rural}, insurance: {insur}\n"
            f"History: {history}\nSymptoms: {symptoms}\n"
            f"Clinical notes: {clinical_notes}\n\n"
            f"Imaging findings:\n{visual}\n\n"
            "Cross-reference the imaging findings with the clinical data. "
            "List 2–3 key clinical correlations and their significance for "
            "maternal-fetal risk."
        )

        inputs = self._tok(prompt, return_tensors="pt",
                           truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=TOP_P,
                pad_token_id=self._tok.pad_token_id,
            )
        return self._tok.decode(ids[0], skip_special_tokens=True)

    # ── Phase 3: Equity Validation ────────────────────────────────────────────

    def _equity_validation(self, correlation: str, demographics: Dict) -> str:
        race  = demographics.get("race_ethnicity", "unknown")
        insur = demographics.get("insurance", "unknown")
        rural = "rural" if demographics.get("rural") else "urban"

        prompt = (
            f"Patient demographics: {race}, {rural}, insurance: {insur}\n"
            f"Clinical correlation:\n{correlation}\n\n"
            "Identify any social determinants of health (SDOH) that may affect "
            "access to care or outcome for this patient. "
            "Suggest 1–2 equity-aware recommendations."
        )

        inputs = self._tok(prompt, return_tensors="pt",
                           truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=TOP_P,
                pad_token_id=self._tok.pad_token_id,
            )
        return self._tok.decode(ids[0], skip_special_tokens=True)

    # ── Phase 4: Risk Stratification ──────────────────────────────────────────

    def _risk_stratification(
        self,
        visual:         str,
        correlation:    str,
        equity:         str,
        clinical_notes: str,
        history:        str,
        symptoms:       str,
        demographics:   Dict,
    ) -> ClinicalAssessment:
        prompt = (
            f"Based on:\n"
            f"• Imaging: {visual[:300]}\n"
            f"• Clinical: {correlation[:300]}\n"
            f"• Equity: {equity[:200]}\n\n"
            "Provide a final risk assessment.\n"
            "Respond EXACTLY in this format:\n"
            "RISK: HIGH|MODERATE|LOW\n"
            "PLANE: <identified plane>\n"
            "CONFIDENCE: high|medium|low\n"
            "REASONING: <2–3 sentences>\n"
            "RECOMMENDATION: <1–2 sentences>"
        )

        inputs = self._tok(prompt, return_tensors="pt",
                           truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=TOP_P,
                pad_token_id=self._tok.pad_token_id,
            )
        raw = self._tok.decode(ids[0], skip_special_tokens=True)
        return self._parse_assessment(raw, visual, correlation, equity)

    # ── Parsing helper ────────────────────────────────────────────────────────

    def _parse_assessment(
        self,
        raw:         str,
        visual:      str,
        correlation: str,
        equity:      str,
    ) -> ClinicalAssessment:
        def extract(pattern, default):
            m = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else default

        risk        = extract(r"RISK:\s*(HIGH|MODERATE|LOW)", "MODERATE")
        plane       = extract(r"PLANE:\s*(.+?)(?:\n|$)", "Unknown")
        conf_word   = extract(r"CONFIDENCE:\s*(high|medium|low)", "medium")
        reasoning   = extract(r"REASONING:\s*(.+?)(?:RECOMMENDATION:|$)", raw[-200:])
        recommend   = extract(r"RECOMMENDATION:\s*(.+?)$", "Further evaluation recommended.")

        return ClinicalAssessment(
            risk_level            = risk,
            risk_score            = RISK_SCORE_MAP.get(risk, 0.55),
            plane_identified      = plane,
            visual_findings       = visual[:400],
            clinical_correlations = correlation[:400],
            equity_notes          = equity[:300],
            reasoning             = reasoning[:400],
            recommendation        = recommend[:300],
            confidence_score      = CONF_MAP.get(conf_word.lower(), 0.60),
            image_used            = self.multimodal,
        )

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

import json
import re
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
from pydantic import BaseModel, ConfigDict, ValidationError

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
    differential_diagnoses: str = ""
    uncertainty_summary:    str = ""
    safety_flags:           str = ""
    review_required:        bool = False
    cannot_assess:          bool = False


class StructuredAssessment(BaseModel):
    """Validated structured schema for final risk output."""

    model_config = ConfigDict(extra="ignore")

    risk_level: Literal["HIGH", "MODERATE", "LOW"]
    plane_identified: str
    confidence: Literal["high", "medium", "low"]
    reasoning: str
    recommendation: str
    differential_diagnoses: list[str] = []
    uncertainty: Dict[str, str] = {}
    review_required: bool = False
    cannot_assess: bool = False


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
        fast_mode:       bool = False,
    ) -> ClinicalAssessment:
        demographics = demographics or {}
        visual  = self._visual_perception(image_path, fast_mode=fast_mode)
        corr    = self._clinical_correlation(visual, clinical_notes,
                                              patient_history, symptoms, demographics,
                                              fast_mode=fast_mode)
        equity  = "Skipped in fast mode." if fast_mode else self._equity_validation(
            corr, demographics, fast_mode=fast_mode
        )
        return self._risk_stratification(visual, corr, equity,
                                         clinical_notes, patient_history,
                                         symptoms, demographics,
                                         fast_mode=fast_mode)

    # ── Phase 1: Visual Perception ────────────────────────────────────────────

    def _visual_perception(self, image_path: str, fast_mode: bool = False) -> str:
        img = self.preprocessor(image_path)

        prompt = (
            "You are reviewing one fetal ultrasound image.\n"
            "Return concise findings in this exact format:\n"
            "PLANE: <fetal brain|fetal abdomen|fetal femur|fetal thorax|maternal cervix|other>\n"
            "FINDINGS: <1-2 short sentences on visible structures>\n"
            "ABNORMALITIES: <none or brief concern>."
        )

        if self.multimodal:
            if hasattr(self.processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                mm_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                mm_prompt = f"<image>\n{prompt}"

            inputs = self.processor(
                text=[mm_prompt], images=[img], return_tensors="pt", padding=True
            ).to(self.device)
        else:
            inputs = self._tok(prompt, return_tensors="pt").to(self.device)

        return self._generate_text(
            inputs=inputs,
            max_new_tokens=96 if fast_mode else MAX_NEW_TOKENS,
            do_sample=False if fast_mode else True,
            temperature=None if fast_mode else TEMPERATURE,
            top_p=None if fast_mode else TOP_P,
        )

    # ── Phase 2: Clinical Correlation ─────────────────────────────────────────

    def _clinical_correlation(
        self,
        visual:         str,
        clinical_notes: str,
        history:        str,
        symptoms:       str,
        demographics:   Dict,
        fast_mode:      bool = False,
    ) -> str:
        age   = demographics.get("age", "unknown")
        insur = demographics.get("insurance", "unknown")
        rural = "rural" if demographics.get("rural") else "urban"

        prompt = (
            f"Patient: {age}y, {rural}, insurance: {insur}\n"
            f"History: {history}\nSymptoms: {symptoms}\n"
            f"Clinical notes: {clinical_notes}\n\n"
            f"Imaging findings:\n{visual}\n\n"
            "Cross-reference imaging with clinical context. "
            "List exactly 3 numbered correlations and a one-line significance for each."
        )

        inputs = self._tok(prompt, return_tensors="pt",
                           truncation=True, max_length=2048).to(self.device)
        return self._generate_text(
            inputs=inputs,
            max_new_tokens=120 if fast_mode else 300,
            do_sample=False if fast_mode else True,
            temperature=None if fast_mode else TEMPERATURE,
            top_p=None if fast_mode else TOP_P,
        )

    # ── Phase 3: Equity Validation ────────────────────────────────────────────

    def _equity_validation(self, correlation: str, demographics: Dict, fast_mode: bool = False) -> str:
        race  = demographics.get("race_ethnicity", "unknown")
        insur = demographics.get("insurance", "unknown")
        rural = "rural" if demographics.get("rural") else "urban"

        prompt = (
            f"Patient demographics: {race}, {rural}, insurance: {insur}\n"
            f"Clinical correlation:\n{correlation}\n\n"
            "Identify any social determinants of health (SDOH) that may affect "
            "access to care or outcome for this patient. "
            "Return exactly 2 short equity-aware recommendations."
        )

        inputs = self._tok(prompt, return_tensors="pt",
                           truncation=True, max_length=2048).to(self.device)
        return self._generate_text(
            inputs=inputs,
            max_new_tokens=64 if fast_mode else 200,
            do_sample=False if fast_mode else True,
            temperature=None if fast_mode else TEMPERATURE,
            top_p=None if fast_mode else TOP_P,
        )

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
        fast_mode:      bool = False,
    ) -> ClinicalAssessment:
        safety_flags, hard_escalate = self._safety_guardrails(
            visual=visual,
            correlation=correlation,
            clinical_notes=clinical_notes,
            symptoms=symptoms,
        )

        data_completeness = self._assess_data_completeness(
            clinical_notes=clinical_notes,
            history=history,
            symptoms=symptoms,
        )
        image_quality = self._assess_image_quality(visual, image_present=bool(self.multimodal))

        prompt = (
            "You are generating a structured maternal-fetal triage decision.\n"
            f"Imaging summary: {visual[:320]}\n"
            f"Clinical summary: {correlation[:320]}\n"
            f"Equity summary: {equity[:220]}\n\n"
            f"Safety signals: {safety_flags or 'none'}\n"
            f"Image quality estimate: {image_quality}\n"
            f"Data completeness estimate: {data_completeness}\n\n"
            "Use demographics only as access/care-planning context, never as direct biological risk evidence. "
            "Use guideline-style triage language (routine follow-up vs expedited specialist review vs urgent escalation).\n\n"
            "Return ONLY valid JSON with these keys and allowed values:\n"
            "{\n"
            '  "risk_level": "HIGH|MODERATE|LOW",\n'
            '  "plane_identified": "short plane label",\n'
            '  "confidence": "high|medium|low",\n'
            '  "reasoning": "2 concise sentences",\n'
            '  "recommendation": "1-2 concise sentences",\n'
            '  "differential_diagnoses": ["dx1 - for: ...; against: ...", "dx2 - for: ...; against: ...", "dx3 - for: ...; against: ..."],\n'
            '  "uncertainty": {"model_confidence": "high|medium|low", "image_quality_confidence": "high|medium|low", "data_completeness_confidence": "high|medium|low"},\n'
            '  "review_required": true|false,\n'
            '  "cannot_assess": true|false\n'
            "}\n"
            "No markdown. No extra keys."
        )

        inputs = self._tok(prompt, return_tensors="pt",
                           truncation=True, max_length=2048).to(self.device)
        raw = self._generate_text(
            inputs=inputs,
            max_new_tokens=140 if fast_mode else 300,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        return self._parse_assessment(
            raw=raw,
            visual=visual,
            correlation=correlation,
            equity=equity,
            safety_flags=safety_flags,
            hard_escalate=hard_escalate,
            image_quality=image_quality,
            data_completeness=data_completeness,
        )

    def _safety_guardrails(
        self,
        visual: str,
        correlation: str,
        clinical_notes: str,
        symptoms: str,
    ) -> tuple[list[str], bool]:
        text = " ".join([visual, correlation, clinical_notes, symptoms]).lower()
        flag_map = {
            "severe ventriculomegaly": "Possible severe ventriculomegaly",
            "hydrops": "Possible hydrops fetalis",
            "placental abruption": "Possible placental abruption",
            "absent fetal cardiac": "Possible absent fetal cardiac activity",
            "no fetal heartbeat": "Possible absent fetal heartbeat",
            "preterm labor": "Preterm labor concern",
            "major anomaly": "Possible major fetal anomaly",
            "restricted growth": "Possible fetal growth restriction",
        }
        flags = [label for needle, label in flag_map.items() if needle in text]
        return flags, len(flags) > 0

    def _assess_image_quality(self, visual: str, image_present: bool) -> str:
        if not image_present:
            return "low"
        text = visual.lower()
        low_markers = [
            "poor image quality",
            "limited visualization",
            "nondiagnostic",
            "artifact",
            "cannot determine",
        ]
        medium_markers = ["partially visualized", "suboptimal", "limited"]
        if any(marker in text for marker in low_markers):
            return "low"
        if any(marker in text for marker in medium_markers):
            return "medium"
        return "high"

    def _assess_data_completeness(self, clinical_notes: str, history: str, symptoms: str) -> str:
        completeness = 0
        if clinical_notes.strip():
            completeness += 1
        if history.strip():
            completeness += 1
        if symptoms.strip():
            completeness += 1

        context_text = f"{clinical_notes} {history} {symptoms}".lower()
        has_gestational_age = bool(re.search(r"\b\d{1,2}\s*(week|wk|wks)\b", context_text))
        has_prior_scan = "prior" in context_text or "previous" in context_text

        if completeness <= 1:
            return "low"
        if completeness == 2 or not has_gestational_age:
            return "medium"
        return "high" if has_prior_scan else "medium"

    def _format_uncertainty(self, model_conf: str, image_quality: str, data_completeness: str) -> str:
        return (
            f"Model confidence: {model_conf}; "
            f"Image quality confidence: {image_quality}; "
            f"Data completeness confidence: {data_completeness}"
        )

    def _generate_text(
        self,
        inputs,
        max_new_tokens: int,
        do_sample: bool,
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> str:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tok.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature if temperature is not None else TEMPERATURE
            gen_kwargs["top_p"] = top_p if top_p is not None else TOP_P

        with torch.no_grad():
            ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_ids = ids[0][prompt_len:]
        if new_ids.numel() == 0:
            return self._tok.decode(ids[0], skip_special_tokens=True).strip()
        return self._tok.decode(new_ids, skip_special_tokens=True).strip()

    # ── Parsing helper ────────────────────────────────────────────────────────

    def _parse_assessment(
        self,
        raw:         str,
        visual:      str,
        correlation: str,
        equity:      str,
        safety_flags: list[str],
        hard_escalate: bool,
        image_quality: str,
        data_completeness: str,
    ) -> ClinicalAssessment:
        structured = self._parse_structured_assessment(raw)
        if structured is not None:
            risk_level = structured.risk_level
            model_conf = structured.uncertainty.get("model_confidence", structured.confidence).lower()
            image_conf = structured.uncertainty.get("image_quality_confidence", image_quality).lower()
            data_conf = structured.uncertainty.get("data_completeness_confidence", data_completeness).lower()

            cannot_assess = bool(structured.cannot_assess) or image_conf == "low"
            review_required = bool(structured.review_required) or hard_escalate or cannot_assess

            if hard_escalate:
                risk_level = "HIGH"
            if cannot_assess:
                risk_level = "MODERATE"

            recommendation = structured.recommendation
            if cannot_assess:
                recommendation = (
                    "Image/data quality is insufficient for a reliable diagnosis; repeat targeted ultrasound and specialist review are recommended."
                )
            elif hard_escalate:
                recommendation = (
                    "Urgent specialist evaluation is recommended due to identified red-flag features."
                )

            return ClinicalAssessment(
                risk_level=risk_level,
                risk_score=RISK_SCORE_MAP.get(risk_level, 0.55),
                plane_identified=structured.plane_identified[:80],
                visual_findings=visual[:400],
                clinical_correlations=correlation[:400],
                equity_notes=equity[:300],
                reasoning=structured.reasoning[:400],
                recommendation=recommendation[:300],
                confidence_score=CONF_MAP.get(model_conf, 0.60),
                image_used=self.multimodal,
                differential_diagnoses="\n".join(structured.differential_diagnoses[:3]),
                uncertainty_summary=self._format_uncertainty(model_conf, image_conf, data_conf),
                safety_flags="; ".join(safety_flags) if safety_flags else "None",
                review_required=review_required,
                cannot_assess=cannot_assess,
            )

        def extract(pattern, default):
            m = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else default

        risk        = extract(r"RISK:\s*(HIGH|MODERATE|LOW)", "MODERATE")
        plane       = extract(r"PLANE:\s*(.+?)(?:\n|$)", "Unknown")
        conf_word   = extract(r"CONFIDENCE:\s*(high|medium|low)", "medium")
        reasoning   = extract(r"REASONING:\s*(.+?)(?:RECOMMENDATION:|$)", raw[-200:])
        recommend   = extract(r"RECOMMENDATION:\s*(.+?)$", "Further evaluation recommended.")
        final_risk = "HIGH" if hard_escalate else risk
        cannot_assess = image_quality == "low"
        if cannot_assess:
            final_risk = "MODERATE"
            recommend = "Image/data quality is insufficient for reliable diagnosis; repeat scan and specialist review recommended."

        return ClinicalAssessment(
            risk_level            = final_risk,
            risk_score            = RISK_SCORE_MAP.get(final_risk, 0.55),
            plane_identified      = plane,
            visual_findings       = visual[:400],
            clinical_correlations = correlation[:400],
            equity_notes          = equity[:300],
            reasoning             = reasoning[:400],
            recommendation        = recommend[:300],
            confidence_score      = CONF_MAP.get(conf_word.lower(), 0.60),
            image_used            = self.multimodal,
            differential_diagnoses = "",
            uncertainty_summary    = self._format_uncertainty(conf_word.lower(), image_quality, data_completeness),
            safety_flags           = "; ".join(safety_flags) if safety_flags else "None",
            review_required        = hard_escalate or cannot_assess,
            cannot_assess          = cannot_assess,
        )

    def _parse_structured_assessment(self, raw: str) -> Optional[StructuredAssessment]:
        raw = raw.strip()
        candidates = [raw]

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.insert(0, raw[start : end + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                return StructuredAssessment.model_validate(payload)
            except (json.JSONDecodeError, ValidationError):
                continue
        return None

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class AgenticTranslator:
    """
    Generates a structured, factual reasoning report grounded in classification data.

    Zero hallucination: All claims are anchored to actual classification results
    (final_label, confidence, signal_analysis, leans_toward).

    Fast: Uses template-based reasoning, no heavy LLM call.
    Clear: Bolds key signals and findings for frontend highlighting.
    """

    def __init__(self, qwen_model: Any = None, qwen_tokenizer: Any = None):
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        # Qwen kept as optional for future lightweight reasoning tasks
        # but NOT used for report generation (template-based instead)

    def generate_report(
        self,
        heatmap_data: List[Dict[str, Any]],
        classification: str,
        confidence: float,
        signal_analysis: Optional[Dict[str, Any]] = None,
        final_label: Optional[str] = None,
        leans_toward: Optional[str] = None,
    ) -> str:
        """
        Generate a structured reasoning report.

        Args:
            heatmap_data: Token-level importance scores
            classification: Model's predicted class ("AI-Generated" or "Human-Written")
            confidence: Confidence percentage (0-100)
            signal_analysis: CSS-v2 stylometric conflict signals
            final_label: User-facing label (may be "Mixed" if conflict)
            leans_toward: Direction if final_label is "Mixed"

        Returns:
            Markdown-formatted report with bold highlights and clear reasoning.
        """

        # =========================================================
        # 1. Extract Key Semantic Triggers (Top Attention Tokens)
        # =========================================================
        valid_words = [item for item in heatmap_data if not item.get("is_noise", True)]
        sorted_words = sorted(
            valid_words, key=lambda x: x.get("normalized_score", 0.0), reverse=True
        )

        # Top 3-4 words that influenced the decision
        top_words = [f"**{item['display_token']}**" for item in sorted_words[:4]]

        if len(top_words) >= 2:
            top_words_str = ", ".join(top_words[:-1]) + f", and {top_words[-1]}"
        elif len(top_words) == 1:
            top_words_str = top_words[0]
        else:
            top_words_str = "structural patterns"

        semantic_anchor = f"The model focused on words like {top_words_str}."

        # =========================================================
        # 2. Determine Final Verdict and Confidence Level
        # =========================================================
        final_verdict = final_label if final_label else classification
        confidence_level = (
            "very high"
            if confidence >= 90
            else (
                "high"
                if confidence >= 70
                else "moderate" if confidence >= 50 else "low"
            )
        )

        # =========================================================
        # 3. Extract Stylometric Signals (CSS-v2)
        # =========================================================
        if not signal_analysis:
            # No stylometric analysis available
            return (
                f"**Verdict:** This text is most likely **{final_verdict}** "
                f"(confidence: {confidence:.0f}%). {semantic_anchor}\n\n"
                f"The stylometric analysis was unavailable, so this assessment is based "
                f"on semantic and structural patterns alone. Review recommended for critical decisions."
            )

        conflict_level = signal_analysis.get("conflict_level", "N/A")
        deberta_direction = signal_analysis.get("deberta_direction", "Unknown")
        xgboost_direction = signal_analysis.get("xgboost_direction", "Unknown")
        ai_signals = signal_analysis.get("shap_ai_signals", [])
        human_signals = signal_analysis.get("shap_human_signals", [])
        css_score = signal_analysis.get("css_conflict_score")
        word_count = signal_analysis.get("word_count", 0)

        # Format signals with bold
        def format_signals(sigs: list) -> str:
            if not sigs:
                return "general phrasing"
            if len(sigs) == 1:
                return f"**{sigs[0].lower()}**"
            return (
                ", ".join(f"**{s.lower()}**" for s in sigs[:-1])
                + f", and **{sigs[-1].lower()}**"
            )

        ai_sigs_str = format_signals(ai_signals)
        human_sigs_str = format_signals(human_signals)

        # =========================================================
        # 4. Build Scenario-Based Reasoning (No Hallucination)
        # =========================================================

        # Scenario: HIGH conflict → Mixed label
        if conflict_level == "HIGH":
            if deberta_direction == "AI":
                return (
                    f"**Verdict:** **Mixed** (leans toward **AI-Generated**)\n\n"
                    f"The semantic analysis suggests **AI-written content**, scoring {confidence:.0f}% confident. "
                    f"{semantic_anchor}\n\n"
                    f"However, the writing style shows **conflicting signals**: {human_sigs_str} appear frequently, "
                    f"which are typical of human writing. This suggests the text might be **AI-generated but with human-like touches**, "
                    f"or possibly **AI-assisted human writing**.\n\n"
                    f"**Recommendation:** Review this text carefully. Highlight the conflicting patterns to determine if human assistance was involved."
                )
            else:  # deberta_direction == "Human"
                return (
                    f"**Verdict:** **Mixed** (leans toward **Human-Written**)\n\n"
                    f"The semantic analysis suggests **human-written content**, scoring {confidence:.0f}% confident. "
                    f"{semantic_anchor}\n\n"
                    f"However, the writing style shows **AI-like patterns**: {ai_sigs_str}. This suggests the text might be **human-written but with AI-assisted editing**, "
                    f"or possibly **AI-generated with human refinement**.\n\n"
                    f"**Recommendation:** Review this text carefully. The presence of AI writing patterns indicates possible AI involvement in composition or editing."
                )

        # Scenario: LOW/MODERATE conflict + Strong agreement
        elif conflict_level in ["LOW", "MODERATE"]:
            if deberta_direction == "AI" and xgboost_direction == "AI":
                return (
                    f"**Verdict:** **AI-Generated** (confidence: {confidence_level})\n\n"
                    f"Both semantic and stylometric analysis agree: this is **AI-written content**. "
                    f"{semantic_anchor}\n\n"
                    f"The writing style confirms this with patterns typical of AI: {ai_sigs_str}. "
                    f"The strong agreement between deep semantic analysis and surface-level style makes this assessment **highly reliable**."
                )
            elif deberta_direction == "Human" and xgboost_direction == "Human":
                return (
                    f"**Verdict:** **Human-Written** (confidence: {confidence_level})\n\n"
                    f"Both semantic and stylometric analysis agree: this is **authentic human writing**. "
                    f"{semantic_anchor}\n\n"
                    f"The writing style confirms this with patterns typical of humans: {human_sigs_str}. "
                    f"The strong agreement between deep semantic analysis and surface-level style makes this assessment **highly reliable**."
                )
            else:
                # Partial agreement
                return (
                    f"**Verdict:** **{final_verdict}** (confidence: {confidence_level})\n\n"
                    f"Semantic analysis suggests **{deberta_direction}-generated**, while stylometric patterns lean toward **{xgboost_direction}**. "
                    f"{semantic_anchor}\n\n"
                    f"The two signals show partial disagreement: semantic patterns include {ai_sigs_str}, while style indicators include {human_sigs_str}. "
                    f"**Recommendation:** The moderate confidence suggests manual review may be warranted."
                )

        # Scenario: N/A or Unknown
        else:
            return (
                f"**Verdict:** **{final_verdict}** (confidence: {confidence:.0f}%)\n\n"
                f"Based on semantic analysis, this text is most likely **{final_verdict}**. "
                f"{semantic_anchor}\n\n"
                f"Stylometric analysis was insufficient (text likely too short or complex), so the verdict relies primarily on "
                f"semantic and structural patterns. The assessment has {confidence_level} confidence."
            )

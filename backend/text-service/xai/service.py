import logging
import time
from typing import Dict, Any, Optional

from .core_extractor import HybridFusionExtractor
from .linguistic_filter import LinguisticNoiseFilter
from .llm_translator import AgenticTranslator

logger = logging.getLogger(__name__)


class XAIService:
    """
    Orchestrates the Explainable AI (XAI) pipeline.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        qwen_model: Any = None,
        qwen_tokenizer: Any = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer

        self.extractor = HybridFusionExtractor(
            model=self.model, tokenizer=self.tokenizer
        )
        self.linguistic_filter = LinguisticNoiseFilter()
        self.translator = AgenticTranslator(
            qwen_model=self.qwen_model, qwen_tokenizer=self.qwen_tokenizer
        )

    def generate_audit(
        self,
        text: str,
        predicted_class: str,
        confidence: float,
        signal_analysis: Optional[Dict[str, Any]] = None,
        telemetry: Optional[Dict[str, float]] = None,
        final_label: Optional[str] = None,
        leans_toward: Optional[str] = None,
    ) -> Dict[str, Any]:

        logger.info("Starting XAI Audit pipeline...")

        if telemetry is None:
            telemetry = {}

        try:
            target_class_idx = 1 if "AI" in predicted_class.upper() else 0

            # Step 1: Core Mathematical Extraction
            t_math = time.time()
            logger.info(
                f"Extracting Hybrid Fusion tensors targeting class index {target_class_idx}..."
            )
            raw_scores, raw_tokens = self.extractor.extract_scores(
                text, target_class_idx=target_class_idx
            )

            # Step 2: Linguistic Noise Filtration
            heatmap_data = self.linguistic_filter.process_and_normalize(
                raw_tokens, raw_scores
            )
            telemetry["math_extraction_time_ms"] = (time.time() - t_math) * 1000

            # Step 3: Agentic LLM Translation
            t_llm = time.time()
            logger.info("Generating unified interpretation report...")
            report = self.translator.generate_report(
                heatmap_data=heatmap_data,
                classification=predicted_class,
                confidence=confidence,
                signal_analysis=signal_analysis,
                final_label=final_label,
                leans_toward=leans_toward,
            )
            telemetry["llm_generation_time_ms"] = (time.time() - t_llm) * 1000

            # Step 4: Construct final response
            return {
                "status": "success",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "heatmap_data": heatmap_data,
                "interpretability_report": report,
            }

        except Exception as e:
            logger.error(f"Error during XAI pipeline execution: {str(e)}")
            raise e

import gc
import numpy as np
import torch
from typing import Tuple, List, Any

try:
    from captum.attr import LayerIntegratedGradients
except ImportError:
    raise ImportError(
        "The 'captum' library is required for Hybrid Fusion. Please install it using 'pip install captum'."
    )


class HybridFusionExtractor:
    """
    Research-Grade XAI Extractor.
    Calculates robust attribution scores by fusing normalized attention weights
    with normalized layer integrated gradients.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: Any, ig_steps: int = 10):
        self.model = model
        self.tokenizer = tokenizer
        self.ig_steps = ig_steps

        if not getattr(self.model.config, "output_attentions", False):
            raise ValueError(
                "Model configuration must have 'output_attentions=True' to utilize this extractor."
            )

        self.device = next(self.model.parameters()).device

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        min_val = np.min(array)
        max_val = np.max(array)
        if max_val - min_val == 0:
            return array
        return (array - min_val) / (max_val - min_val)

    def _get_base_attention(self, inputs: dict) -> np.ndarray:
        """Extracts averaged [CLS] attention weights from the final transformer layer."""
        with torch.no_grad():
            outputs = self.model(**inputs)

        if outputs.attentions is None:
            raise RuntimeError("Model did not return attentions.")

        # final_layer_attention: (batch_size, num_heads, seq_len, seq_len)
        final_layer_attention = outputs.attentions[-1]
        cls_attention = final_layer_attention[0, :, 0, :]
        avg_attention = cls_attention.mean(dim=0).cpu().numpy()

        return avg_attention

    def _get_integrated_gradients(
        self, inputs: dict, target_class_idx: int
    ) -> np.ndarray:
        """Extracts Integrated Gradients targeting the word embeddings layer with low VRAM footprint."""

        def custom_forward(input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

        # Dynamic embedding layer discovery across Hugging Face architectures
        if hasattr(self.model, "deberta_v2"):
            word_embeddings_layer = self.model.deberta_v2.embeddings.word_embeddings
        elif hasattr(self.model, "deberta"):
            word_embeddings_layer = self.model.deberta.embeddings.word_embeddings
        elif hasattr(self.model, "roberta"):
            word_embeddings_layer = self.model.roberta.embeddings.word_embeddings
        elif hasattr(self.model, "bert"):
            word_embeddings_layer = self.model.bert.embeddings.word_embeddings
        elif hasattr(self.model, "get_input_embeddings"):
            word_embeddings_layer = self.model.get_input_embeddings()
        else:
            raise ValueError(
                "Unsupported architecture: cannot locate word embeddings layer."
            )

        lig = LayerIntegratedGradients(custom_forward, word_embeddings_layer)

        # Baseline: [PAD] or 0 token tensors of identical shape
        baseline_id = self.tokenizer.pad_token_id or 0
        baselines = torch.full_like(inputs["input_ids"], baseline_id)

        try:
            attributions = lig.attribute(
                inputs=inputs["input_ids"],
                baselines=baselines,
                additional_forward_args=(inputs["attention_mask"],),
                target=target_class_idx,
                n_steps=self.ig_steps,
                internal_batch_size=1,  # Keeps VRAM consumption minimal (avoids OOM on 15GB T4)
                return_convergence_delta=False,
            )

            attributions_sum = attributions.sum(dim=-1).squeeze(0).cpu().detach()
            ig_scores = np.abs(attributions_sum.numpy())
            return ig_scores

        finally:
            # Explicit cleanup of transient attribution tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def extract_scores(
        self, text: str, target_class_idx: int = 1
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Executes the full Hybrid Fusion pipeline.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        raw_attention = self._get_base_attention(inputs)
        raw_ig = self._get_integrated_gradients(inputs, target_class_idx)

        norm_attention = self._normalize(raw_attention)
        norm_ig = self._normalize(raw_ig)

        fused_scores = norm_attention * norm_ig
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return fused_scores, tokens

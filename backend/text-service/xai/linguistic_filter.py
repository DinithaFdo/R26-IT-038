import spacy
import numpy as np
import subprocess
from typing import List, Dict, Any


class LinguisticNoiseFilter:
    """
    Applies NLP techniques to interpretability tensors to isolate true semantic
    causal triggers by zeroing out structural and grammatical noise.
    
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initializes the spaCy NLP engine and defines the noise filtering parameters.
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            # Fallback to download the model dynamically if missing in the environment/Docker
            subprocess.run(
                ["python", "-m", "spacy", "download", spacy_model], check=True
            )
            self.nlp = spacy.load(spacy_model)

        # Define the exact POS tags we want to MASK (turn to 0)
        # PUNCT (Punctuation), CCONJ (Conjunctions), DET (Determiners),
        # ADP (Adpositions), SPACE (Whitespace), PART (Particles)
        self.noise_pos_tags = {"PUNCT", "CCONJ", "DET", "ADP", "SPACE", "PART"}

        # Transformer-specific structural tokens to ignore
        self.structural_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}

    def process_and_normalize(
        self, raw_tokens: List[str], hybrid_scores: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Cleans tokens, merges subwords, applies POS masking, and prepares the final UI-ready data structure.
        """
        import string  # Required for punctuation detection

        # =========================================================
        # STEP 1: MERGE SUBWORDS (e.g., "Din" + "itha" -> "Dinitha")
        # =========================================================
        merged_tokens = []
        current_word = ""
        current_original = ""
        current_scores = []

        for token, score in zip(raw_tokens, hybrid_scores):
            # 1a. Handle Structural Tokens (Immediate flush)
            if token in self.structural_tokens:
                if current_word:
                    merged_tokens.append(
                        (current_original, current_word, float(np.mean(current_scores)))
                    )
                    current_word, current_original, current_scores = "", "", []
                merged_tokens.append((token, token, float(score)))
                continue

            # 1b. Clean the token for analysis
            clean_piece = (
                token.replace("Ġ", "")
                .replace(" ", "")
                .replace("_", "")
                .replace("\u2581", "")
                .replace("▁", "")
                .strip()
            )

            # 1c. Determine if this token is the START of a new word
            # DeBERTa uses specific characters for new words. Also, isolate pure punctuation.
            is_prefix = token.startswith(("Ġ", " ", "\u2581", "_", "▁"))
            is_punct = (
                all(char in string.punctuation for char in clean_piece)
                if clean_piece
                else False
            )

            is_new_word = is_prefix or is_punct

            # If it's a new word OR our current word buffer is empty
            if is_new_word or not current_word:
                # Flush the previous word if it exists
                if current_word:
                    merged_tokens.append(
                        (current_original, current_word, float(np.mean(current_scores)))
                    )

                # Start a new word buffer
                current_word = clean_piece
                current_original = token
                current_scores = [float(score)]
            else:
                # It's a subword (like "itha")! Append it to the current word ("Din")
                current_word += clean_piece
                current_original += token
                current_scores.append(float(score))

        # Flush the final word in the buffer
        if current_word:
            merged_tokens.append(
                (current_original, current_word, float(np.mean(current_scores)))
            )

        # =========================================================
        # STEP 2: POS MASKING & NOISE FILTRATION
        # =========================================================
        processed_data = []
        valid_scores = []

        for orig_tok, disp_tok, avg_score in merged_tokens:
            is_noise = False

            # Mask structural tokens or empty strings
            if orig_tok in self.structural_tokens or not disp_tok:
                is_noise = True
            else:
                # Mask grammatical noise using spaCy
                # We lower() the token to ensure spaCy's out-of-context tagger doesn't misclassify capital letters
                spacy_doc = self.nlp(disp_tok.lower())
                if len(spacy_doc) > 0:
                    spacy_token = spacy_doc[0]
                    # UPGRADE: Added `spacy_token.is_stop` to catch all auxiliary verbs (is, are)
                    # and guarantee strict removal of all English stop words (a, the, of).
                    if (
                        spacy_token.pos_ in self.noise_pos_tags
                        or spacy_token.is_punct
                        or spacy_token.is_stop
                    ):
                        is_noise = True

            processed_data.append(
                {
                    "original_token": orig_tok,
                    "display_token": disp_tok,
                    "is_noise": is_noise,
                    "raw_score": avg_score,
                }
            )

            if not is_noise:
                valid_scores.append(avg_score)

        # =========================================================
        # STEP 3: NORMALIZATION FOR UI HEATMAP
        # =========================================================
        if valid_scores:
            min_score = min(valid_scores)
            max_score = max(valid_scores)
            range_score = max_score - min_score if max_score > min_score else 1.0

            for item in processed_data:
                if item["is_noise"]:
                    item["normalized_score"] = 0.0  # Force noise to be invisible
                else:
                    item["normalized_score"] = (
                        item["raw_score"] - min_score
                    ) / range_score
        else:
            # Edge-case fallback
            for item in processed_data:
                item["normalized_score"] = 0.0

        return processed_data

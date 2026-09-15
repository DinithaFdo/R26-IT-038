import inspect
import logging
import math
import time
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from classification.model_loader import get_tokenizer, get_model, get_device, MAX_LENGTH
from classification.sanitization import sanitize
from shared.schemas import ClassifyResponse, TokenHighlight, SanitizationResult

import re
import string
import numpy as np
from collections import Counter

import spacy
import shap

from classification.model_loader import (
    get_xgb_model, get_xgb_scaler, get_xgb_feature_names,
    get_xgb_calibrator, get_deberta_temperature,
    get_css_min_word_count, is_css_available,
)
from shared.schemas import SignalAnalysis

# Words excluded from token-highlighting -- they carry little discriminative
# signal and would otherwise dominate the "low importance" bucket.
STOPWORDS = {
    "the", "a", "an", "is", "in", "of", "to", "and",
    "that", "it", "was", "are", "we", "i", "has", "have", "this", "with", "for", "at",
    "be", "by", "as", "on", "not", "or", "but", "its", "our", "they", "which", "who",
}

# ── CSS-v2: Vocabulary sets for feature extraction ──────────────
_FORMAL_WORDS = {
    'utilize','implement','demonstrate','facilitate','establish',
    'leverage','optimize','endeavor','comprehensive','fundamental',
    'significant','substantial','consequently','furthermore',
    'nevertheless','predominantly','subsequent','methodology',
    'framework','paradigm','constitute','infrastructure','mechanism',
    'collaboration','integration','initiative','objective','strategic',
    'deployment','evaluation','ascertain','preliminary',
    'notwithstanding','aforementioned','incorporate','indicate',
    'obtain','provide','require','ensure','maintain','achieve',
    'represent','determine','identify','conduct','develop','propose',
    'examine','emphasize','acknowledge','recognize','address',
    'analyze','analyse','explore','discuss','consider','investigate',
    'highlight','illustrate','suggest','reveal','reflect','conclude',
    'observe','argue','present','describe','compare','evaluate',
    'assess','review','explain','define','interpret','justify',
    'validate','verify','quantify','categorize','classify','measure',
    'monitor','enhance','improve','increase','decrease','affect',
    'impact','influence','contribute','support','enable','allow',
    'prevent','reduce','involve','include','consist','contain',
    'therefore','however','moreover','although','whereas','whereby',
    'thereby','herein','therein','subsequently','additionally',
    'alternatively','specifically','particularly','significantly',
    'substantially','effectively','efficiently','successfully',
    'appropriately','accordingly','comprehensively',
}
_PERSONAL_PRONOUNS = {
    'i','me','my','mine','myself','we','us','our','ours','ourselves'
}
_FIRST_PERSON_SINGULAR = {'i','me','my','mine','myself'}
_THIRD_PERSON = {
    'he','him','his','himself','she','her','hers','herself',
    'it','its','itself','they','them','their','theirs','themselves'
}
_FUNCTION_WORDS = {
    'the','a','an','is','are','was','were','be','been','have','has',
    'had','do','does','did','will','would','could','should','may',
    'might','and','but','or','nor','for','yet','so','at','by',
    'from','in','into','of','on','to','up','with','as','if',
    'though','because',
}
_HEDGE_WORDS = {
    'maybe','perhaps','probably','might','possibly','apparently',
    'seemingly','sort','kind','guess','unclear','uncertain',
}
_MODAL_VERBS = {
    'can','could','would','should','may','might','must','shall','will'
}
_DISCOURSE_MARKERS = {
    'however','therefore','moreover','furthermore','nevertheless',
    'consequently','additionally','alternatively','specifically',
    'particularly','notably','importantly','significantly',
    'interestingly','surprisingly','accordingly','thus','hence',
}
_CONTRACTIONS = re.compile(
    r"\b(don't|doesn't|didn't|won't|wouldn't|can't|couldn't|"
    r"shouldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|"
    r"i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|"
    r"it's|we're|we've|we'll|we'd|they're|they've|they'll|they'd|"
    r"that's|there's|here's|what's|who's|how's|n't)\b",
    re.IGNORECASE,
)
_COORD_CONJ = {'and','but','or','nor','for','yet','so'}
_CONTENT_POS = {'NOUN','VERB','ADJ','ADV'}

_SHAP_DISPLAY_NAMES = {
    'pronoun_ratio':            'Personal pronoun usage',
    'formal_ratio':             'Formal vocabulary density',
    'contraction_ratio':        'Contraction usage',
    'sentence_avg_len':         'Average sentence length',
    'sentence_len_variance':    'Sentence length variation',
    'avg_word_length':          'Average word length',
    'function_word_ratio':      'Function word usage',
    'hapax_ratio':              'Vocabulary uniqueness',
    'punctuation_ratio':        'Punctuation density',
    'passive_ratio':            'Passive voice usage',
    'hedge_ratio':              'Hedging language usage',
    'sttr':                     'Standardised lexical diversity',
    'mtld':                     'Textual lexical diversity',
    'ppl_mean':                 'Text predictability (perplexity)',
    'ppl_std':                  'Perplexity variation',
    'repetition_ratio':         'Phrase repetition density',
    'discourse_marker_density': 'Discourse marker usage',
    'question_ratio':           'Interrogative sentence proportion',
    'exclamation_ratio':        'Exclamation usage',
    'avg_dep_depth':            'Syntactic complexity',
    'lexical_density':          'Content word density',
    'comma_rate':               'Comma usage per sentence',
    'quote_ratio':              'Quotation mark usage',
    'list_density':             'List or bullet point density',
    'char_per_word':            'Character per word ratio',
    'vocab_size_ratio':         'Vocabulary breadth',
    'first_person_singular':    'First-person singular pronoun usage',
    'third_person_ratio':       'Third-person pronoun usage',
    'conjunction_ratio':        'Coordinating conjunction usage',
    'modal_verb_ratio':         'Modal verb usage',
}

# Lazy-loaded spaCy and SHAP (loaded on first CSS call, not at import)
_nlp = None
_shap_explainer = None


def _forward_kwargs(model, encoded):
    """
    Keep only the encoded tensor keys that `model.forward()` actually accepts.

    Some tokenizers emit keys (e.g. token_type_ids) that a given model's
    forward() signature doesn't declare, which raises a TypeError at call
    time. Inspecting the signature lets us safely drop anything unexpected.
    """
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    return {key: value for key, value in encoded.items() if key in forward_params}


def predict(text: str) -> tuple[str, float, float]:
    """
    Run the classifier on `text` and return (label, prob_ai, prob_human).

    label = "AI-Generated" if prob_ai >= 0.5 else "Human-Written"
    """
    tokenizer = get_tokenizer()
    model = get_model()
    device = get_device()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model_inputs = _forward_kwargs(model, encoded)

    # No gradients needed for inference -- saves memory and compute.
    with torch.no_grad():
        outputs = model(**model_inputs)

    # Label indices from model.config.id2label
    # Training convention: 0=Human, 1=AI (verified via Swagger tests)
    id2label = model.config.id2label
    human_idx = next((idx for idx, name in id2label.items() if "human" in name.lower()), 0)
    ai_idx = next((idx for idx, name in id2label.items() if "ai" in name.lower()), 1)

    probs = F.softmax(outputs.logits, dim=-1)[0]
    prob_human = probs[human_idx].item()
    prob_ai = probs[ai_idx].item()

    label = "AI-Generated" if prob_ai >= 0.5 else "Human-Written"
    return label, prob_ai, prob_human


def get_token_highlights(text: str) -> list[TokenHighlight]:
    """
    Compute per-word importance highlights for `text`.

    Importance is derived from the model's self-attention weights: how
    strongly the [CLS] token attends to each other token, averaged across
    every transformer layer and attention head.
    """
    tokenizer = get_tokenizer()
    model = get_model()
    device = get_device()

    # No padding -- single example, and we don't want PAD tokens polluting
    # the attention matrix.
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model_inputs = _forward_kwargs(model, encoded)

    with torch.no_grad():
        outputs = model(**model_inputs, output_attentions=True)

    # outputs.attentions: one (batch, num_heads, seq_len, seq_len) tensor per
    # layer. Stack across layers, then average across the layer and head
    # dimensions to collapse everything into a single (batch, seq, seq)
    # attention matrix. The [CLS] row (index 0) gives each token's overall
    # importance as attended-to by [CLS].
    attn_stack = torch.stack(outputs.attentions, dim=0)  # (layers, batch, heads, seq, seq)
    attn_avg = attn_stack.mean(dim=(0, 2))  # (batch, seq, seq)
    cls_attention = attn_avg[0, 0].tolist()  # (seq,)

    input_ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    special_tokens = set(tokenizer.all_special_tokens)

    # Merge DeBERTa sentencepiece subwords back into whole words. A token
    # starting with "▁" marks the start of a new word; tokens without the
    # prefix are continuations of the previous word and get merged in, with
    # their scores averaged.
    words = []
    current_word = None
    current_scores = []

    for token, score in zip(tokens, cls_attention):
        if token in special_tokens:
            continue

        if token.startswith("▁") or current_word is None:
            if current_word is not None:
                words.append({"word": current_word, "score": sum(current_scores) / len(current_scores)})
            current_word = token.lstrip("▁")
            current_scores = [score]
        else:
            current_word += token
            current_scores.append(score)

    if current_word is not None:
        words.append({"word": current_word, "score": sum(current_scores) / len(current_scores)})

    # Only highlight "content" words -- stopwords and single-character
    # tokens carry little semantic weight and are excluded entirely.
    content_words = [
        w for w in words
        if w["word"].lower() not in STOPWORDS and len(w["word"]) > 1
    ]

    if not content_words:
        return []

    # Min-max normalize raw attention scores to the 0-1 range, computed over
    # content words only so stopwords/short tokens can't skew the scale.
    raw_scores = [w["score"] for w in content_words]
    min_score, max_score = min(raw_scores), max(raw_scores)
    score_span = max_score - min_score

    for w in content_words:
        w["norm_score"] = (w["score"] - min_score) / score_span if score_span > 0 else 0.0

    # Percentile-based bucketing: rank content words by normalized score and
    # split into top 20% ("high"), next 30% ("mid"), remaining 50% ("low").
    ranked = sorted(content_words, key=lambda w: w["norm_score"], reverse=True)
    total = len(ranked)
    high_count = math.ceil(total * 0.2)
    mid_count = math.ceil(total * 0.3)

    for idx, w in enumerate(ranked):
        if idx < high_count:
            w["level"] = "high"
        elif idx < high_count + mid_count:
            w["level"] = "mid"
        else:
            w["level"] = "low"

    # Return in original reading order (not score-sorted order) so callers
    # can render highlights inline with the source text.
    return [
        TokenHighlight(word=w["word"], score=round(w["norm_score"], 4), level=w["level"])
        for w in content_words
    ]


def _get_nlp():
    """Lazy-load spaCy model on first CSS-v2 call."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp


def _compute_sttr(words: list[str], window: int = 50, step: int = 10) -> float:
    if len(words) < window:
        return len(set(words)) / len(words) if words else 0.0
    return float(np.mean([
        len(set(words[i:i + window])) / window
        for i in range(0, len(words) - window + 1, step)
    ]))


def _compute_mtld(words: list[str], threshold: float = 0.72) -> float:
    def _pass(w):
        types, tokens, factors = set(), 0, 0
        for word in w:
            tokens += 1
            types.add(word)
            if len(types) / tokens <= threshold:
                factors += 1
                types, tokens = set(), 0
        if tokens > 0:
            factors += (1 - len(types) / tokens) / (1 - threshold)
        return len(w) / factors if factors > 0 else 0.0
    if len(words) < 10:
        return 0.0
    return (_pass(words) + _pass(words[::-1])) / 2


def _extract_features(text: str, feature_names: list[str]) -> np.ndarray:
    """Extract 30 stylometric features. Returns numpy array ordered by feature_names."""
    tc = text.strip()
    tl = tc.lower()
    nlp = _get_nlp()

    doc_tokens = _get_nlp()(tl)
    raw = [token.text for token in doc_tokens]
    words = [token.text for token in doc_tokens if token.is_alpha]
    nw = len(words) if words else 1

    doc = nlp(tc[:50000])
    sents = list(doc.sents)
    sl = [len(list(s)) for s in sents]
    ns = len(sl) if sl else 1

    wc = Counter(words)
    bg = list(zip(words[:-1], words[1:]))
    bgc = Counter(bg)

    depths = []
    for tok in doc:
        if tok.is_space:
            continue
        d, t = 0, tok
        while t.head != t:
            t = t.head
            d += 1
        depths.append(d)

    feat_map = {
        'pronoun_ratio':            sum(1 for w in words if w in _PERSONAL_PRONOUNS) / nw,
        'formal_ratio':             sum(1 for w in words if w in _FORMAL_WORDS) / nw,
        'contraction_ratio':        len(_CONTRACTIONS.findall(tc)) / nw,
        'sentence_avg_len':         float(np.mean(sl)) if sl else 0.0,
        'sentence_len_variance':    float(np.var(sl)) if len(sl) > 1 else 0.0,
        'avg_word_length':          float(np.mean([len(w) for w in words])) if words else 0.0,
        'function_word_ratio':      sum(1 for w in words if w in _FUNCTION_WORDS) / nw,
        'hapax_ratio':              sum(1 for c in wc.values() if c == 1) / len(wc) if wc else 0.0,
        'punctuation_ratio':        sum(1 for c in tc if c in string.punctuation) / len(tc) if tc else 0.0,
        'passive_ratio':            sum(1 for t in doc if t.dep_ == 'auxpass') / ns,
        'hedge_ratio':              sum(1 for w in words if w in _HEDGE_WORDS) / nw,
        'sttr':                     _compute_sttr(words),
        'mtld':                     _compute_mtld(words),
        'ppl_mean':                 0.0,
        'ppl_std':                  0.0,
        'repetition_ratio':         sum(c for c in bgc.values() if c > 1) / len(bg) if bg else 0.0,
        'discourse_marker_density': sum(1 for w in words if w in _DISCOURSE_MARKERS) / nw,
        'question_ratio':           sum(1 for s in sents if s.text.strip().endswith('?')) / ns,
        'exclamation_ratio':        sum(1 for s in sents if s.text.strip().endswith('!')) / ns,
        'avg_dep_depth':            float(np.mean(depths)) if depths else 0.0,
        'lexical_density':          sum(1 for t in doc if t.pos_ in _CONTENT_POS) / (len([t for t in doc if not t.is_space]) or 1),
        'comma_rate':               tc.count(',') / ns,
        'quote_ratio':              (tc.count('"') + tc.count("'") +
                                     tc.count('“') + tc.count('”')) / nw,
        'list_density':             len(re.findall(r'^\s*[-•*]\s+|^\s*\d+\.\s+', tc, re.MULTILINE)) / (nw / 100 + 1e-9),
        'char_per_word':            float(np.mean([len(t) for t in raw])) if raw else 0.0,
        'vocab_size_ratio':         len(wc) / nw,
        'first_person_singular':    sum(1 for w in words if w in _FIRST_PERSON_SINGULAR) / nw,
        'third_person_ratio':       sum(1 for w in words if w in _THIRD_PERSON) / nw,
        'conjunction_ratio':        sum(1 for w in words if w in _COORD_CONJ) / nw,
        'modal_verb_ratio':         sum(1 for w in words if w in _MODAL_VERBS) / nw,
    }

    arr = np.array([[feat_map.get(f, 0.0) for f in feature_names]])
    return np.nan_to_num(arr, nan=0.0, posinf=5000.0, neginf=-5000.0)


def _get_shap_explainer():
    """Lazy-load SHAP TreeExplainer on first CSS call."""
    global _shap_explainer
    if _shap_explainer is None and is_css_available():
        _shap_explainer = shap.TreeExplainer(get_xgb_model())
    return _shap_explainer


def _get_final_label(
    label: str,
    conflict_level: Optional[str],
) -> tuple[str, Optional[str]]:
    """
    Determines the user-facing final label based
    on DeBERTa result and CSS conflict level.

    Rules:
      HIGH conflict   → "Mixed" (strong disagreement)
      MODERATE/LOW/NA → Keep DeBERTa label unchanged

    Args:
        label: DeBERTa raw decision
               "AI-Generated" or "Human-Written"
        conflict_level: CSS result
                        "HIGH","MODERATE","LOW","N/A",None

    Returns:
        (final_label, leans_toward)
        final_label:  "AI-Generated"|"Human-Written"|"Mixed"
        leans_toward: "AI-Generated"|"Human-Written"|None
    """
    if conflict_level == "HIGH":
        return "Mixed", label
    return label, None


def _compute_signal_analysis(clean_text: str, prob_ai: float) -> SignalAnalysis:

    """
    Compute CSS-v2 signal analysis.
    Returns SignalAnalysis with null CSS fields if text is too short
    or XGBoost artifacts are unavailable.
    """
    word_count = len(clean_text.split())
    deberta_direction = "AI" if prob_ai >= 0.5 else "Human"
    min_words = get_css_min_word_count()

    if word_count < min_words:
        return SignalAnalysis(
            css_conflict_score=None,
            conflict_level="N/A",
            conflict_detected=False,
            deberta_direction=deberta_direction,
            xgboost_direction=None,
            shap_ai_signals=[],
            shap_human_signals=[],
            word_count=word_count,
            note=f"Stylometric analysis requires minimum {min_words} words.",
        )

    if not is_css_available():
        return SignalAnalysis(
            css_conflict_score=None,
            conflict_level="N/A",
            conflict_detected=False,
            deberta_direction=deberta_direction,
            xgboost_direction=None,
            shap_ai_signals=[],
            shap_human_signals=[],
            word_count=word_count,
            note="Stylometric model unavailable.",
        )

    try:
        feature_names = get_xgb_feature_names()
        scaler        = get_xgb_scaler()
        xgb_model     = get_xgb_model()
        calibrator    = get_xgb_calibrator()
        T             = get_deberta_temperature()

        # Feature extraction
        feat_arr   = _extract_features(clean_text, feature_names)
        feat_sc    = scaler.transform(feat_arr)
        p_raw      = xgb_model.predict_proba(feat_sc)[0][1]
        p_style_cal = float(calibrator.predict([p_raw])[0])

        # Temperature-calibrated DeBERTa probability
        import math
        logit_ai    = math.log(max(prob_ai, 1e-9)) / T
        logit_human = math.log(max(1 - prob_ai, 1e-9)) / T
        exp_ai      = math.exp(logit_ai)
        p_sem_cal   = exp_ai / (exp_ai + math.exp(logit_human))

        # Signed margin conflict score
        s_D = 2 * p_sem_cal - 1
        s_S = 2 * p_style_cal - 1
        C   = float(max(0.0, -(s_D * s_S)))

        if C > 0.60:
            conflict_level = "HIGH"
        elif C > 0.30:
            conflict_level = "MODERATE"
        else:
            conflict_level = "LOW"

        # SHAP signals
        explainer  = _get_shap_explainer()
        shap_vals  = explainer.shap_values(feat_sc)[0]
        pairs      = list(zip(feature_names, shap_vals))
        pairs.sort(key=lambda x: x[1], reverse=True)
        ai_signals    = [_SHAP_DISPLAY_NAMES.get(f, f) for f, v in pairs if v > 0][:3]
        human_signals = [_SHAP_DISPLAY_NAMES.get(f, f) for f, v in pairs if v < 0][:3]

        return SignalAnalysis(
            css_conflict_score=round(C, 3),
            conflict_level=conflict_level,
            conflict_detected=C > 0.30,
            deberta_direction=deberta_direction,
            xgboost_direction="AI" if s_S > 0 else "Human",
            shap_ai_signals=ai_signals,
            shap_human_signals=human_signals,
            word_count=word_count,
            note=None,
        )

    except Exception as e:
        logger.warning(f"CSS-v2 signal analysis failed: {e}")
        return SignalAnalysis(
            css_conflict_score=None,
            conflict_level="N/A",
            conflict_detected=False,
            deberta_direction=deberta_direction,
            xgboost_direction=None,
            shap_ai_signals=[],
            shap_human_signals=[],
            word_count=word_count,
            note="Signal analysis encountered an error.",
        )


def classify(text: str, include_highlights: bool = True) -> ClassifyResponse:
    """
    Full classification pipeline: sanitize -> predict -> highlight -> assemble response.

    Note: sanitize() (see classification/sanitization.py) is expected to
    return a dict with keys "clean_text", "was_attacked", "attack_report".
    """
    start = time.time()

    san_result = sanitize(text)
    clean_text = san_result["clean_text"]
    was_attacked = san_result["attacks_detected"]
    attack_report = san_result["attack_report"]

    sanitization_result = SanitizationResult(
        was_attacked=was_attacked,
        attack_report=attack_report,
        clean_text=clean_text,
    )

    # Classify on the sanitized text so adversarial input can't skew the prediction.
    label, prob_ai, prob_human = predict(clean_text)

    # Highlights on clean_text for consistency with classification
    token_highlights = (
        get_token_highlights(clean_text)
        if include_highlights
        else []
    )

    processing_time_ms = (time.time() - start) * 1000

    signal_analysis = _compute_signal_analysis(clean_text, prob_ai)

    # Compute user-facing final label based on CSS
    final_label, leans_toward = _get_final_label(
        label=label,
        conflict_level=(
            signal_analysis.conflict_level
            if signal_analysis is not None
            else None
        ),
    )

    return ClassifyResponse(
        label=label,
        prob_ai=prob_ai,
        prob_human=prob_human,
        sanitization=sanitization_result,
        token_highlights=token_highlights,
        model_version="deberta-v3-large-model2-domainfix",
        processing_time_ms=processing_time_ms,
        signal_analysis=signal_analysis,
        final_label=final_label,
        leans_toward=leans_toward,
    )

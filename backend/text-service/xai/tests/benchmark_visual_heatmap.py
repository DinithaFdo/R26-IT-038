import time
import pandas as pd
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shap
from lime.lime_text import LimeTextExplainer
from transformers import pipeline
from xai.core_extractor import HybridFusionExtractor
from xai.linguistic_filter import LinguisticNoiseFilter
from classification.model_loader import load_models
import classification.model_loader as model_loader


def normalize_array(arr):
    """Min-Max normalization to get scores between 0 and 1."""
    arr = np.array(arr)
    if len(arr) == 0:
        return arr
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def run_visual_benchmark(model, tokenizer, device):
    print("Starting token-level explanation comparison benchmark")
    print("-" * 70)

    # A highly stylometric AI-like sentence for testing
    test_text = "Furthermore, the multifaceted tapestry of this system, which operates seamlessly, is a testament to modern algorithmic design."
    target_class_idx = 1

    # 1. Initialize Tools
    hybrid_extractor = HybridFusionExtractor(
        model=model, tokenizer=tokenizer, ig_steps=10
    )
    noise_filter = LinguisticNoiseFilter()
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1,
    )
    shap_explainer = shap.Explainer(pipe)
    lime_explainer = LimeTextExplainer(class_names=["Human", "AI-Generated"])

    def lime_predict_proba(texts):
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()

    # --- 2. Extract Data ---
    print("Running Hybrid Fusion...")
    raw_scores, raw_tokens = hybrid_extractor.extract_scores(
        test_text, target_class_idx=target_class_idx
    )
    hybrid_data = noise_filter.process_and_normalize(raw_tokens, raw_scores)

    print("Running SHAP...")
    shap_vals = shap_explainer([test_text])
    shap_raw_scores = shap_vals.values[0][:, target_class_idx]
    shap_tokens = shap_vals.data[0]
    shap_norm = normalize_array(shap_raw_scores)

    print("Running LIME...")
    lime_exp = lime_explainer.explain_instance(
        test_text, lime_predict_proba, num_features=100, num_samples=100
    )
    lime_results = dict(lime_exp.as_list())

    # --- 3. Align Data for Plotting ---
    # We use the clean words from Hybrid as our baseline X-axis
    words = []
    hybrid_scores = []
    shap_scores = []
    lime_scores = []

    for i, item in enumerate(hybrid_data):
        word = item["display_token"]
        is_noise = item["is_noise"]

        # Skip empty tokens
        if not word.strip():
            continue

        words.append(word)
        hybrid_scores.append(item.get("normalized_score", 0.0))

        # Find matching SHAP score (approximate matching for subwords)
        shap_score = 0.0
        for s_tok, s_score in zip(shap_tokens, shap_norm):
            clean_s = s_tok.replace("Ġ", "").strip()
            if clean_s and clean_s in word:
                shap_score = max(shap_score, s_score)  # Take max subword score
        shap_scores.append(shap_score)

        # Find matching LIME score
        # LIME normalizes around 0, we take absolute magnitude and re-normalize later
        l_score = lime_results.get(word.lower(), 0.0)
        lime_scores.append(abs(l_score))

    # Normalize LIME
    lime_scores = normalize_array(lime_scores)

    # --- 4. Generate the publication-oriented token-importance heatmap ---
    print("Generating token-importance heatmap...")

    plot_df = pd.DataFrame(
        [hybrid_scores, shap_scores, lime_scores],
        index=["HNIF", "SHAP", "LIME"],
        columns=words,
    )
    informative_columns = plot_df.max(axis=0) > 0.05
    plot_df = plot_df.loc[:, informative_columns]

    sns.set_theme(style="white", context="paper")
    fig_width = max(8.0, min(16.0, 0.42 * len(plot_df.columns)))
    fig, ax = plt.subplots(figsize=(fig_width, 3.8))
    sns.heatmap(
        plot_df,
        cmap="magma",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8, "color": "white"},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Normalized token importance"},
        ax=ax,
    )
    ax.set_title(
        "Token-Level Importance Across Explanation Methods\n"
        "(color intensity and normalized score)",
        pad=12,
    )
    ax.set_xlabel("Input token")
    ax.set_ylabel("Explanation method")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig("heatmap_comparison_plot.png", dpi=300, bbox_inches="tight")
    fig.savefig("heatmap_comparison_plot.pdf", bbox_inches="tight")
    plt.close(fig)

    print(
        "Benchmark complete. Saved figures to 'heatmap_comparison_plot.png' and 'heatmap_comparison_plot.pdf'."
    )


if __name__ == "__main__":
    print("Loading models (this takes a few seconds)...")
    load_models()

    model = None
    tokenizer = None

    for attr_name in dir(model_loader):
        attr_val = getattr(model_loader, attr_name)
        if isinstance(attr_val, torch.nn.Module):
            model = attr_val
        if hasattr(attr_val, "convert_ids_to_tokens"):
            tokenizer = attr_val

    if model is None or tokenizer is None:
        print("❌ Error: Could not automatically detect the model.")
        sys.exit(1)

    actual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(actual_device)

    run_visual_benchmark(model, tokenizer, actual_device)

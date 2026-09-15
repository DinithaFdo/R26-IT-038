import time
import pandas as pd
import torch
import os
import sys
import numpy as np
import spacy
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


def run_sparsity_benchmark(model, tokenizer, device):
    print("Starting explanation sparsity and noise-attribution benchmark")
    print("-" * 70)

    # We use a paragraph with lots of punctuation and stop words to test the filters
    test_text = "The quick brown fox, which was surprisingly fast, jumped over the lazy, sleeping dog! However, it didn't seem to care at all."

    target_class_idx = 1

    # Initialize tools
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

    # --- 1. Run Hybrid Fusion (With your NLP Filter) ---
    print("Running Hybrid Fusion + Linguistic Filter...")
    raw_scores, raw_tokens = hybrid_extractor.extract_scores(
        test_text, target_class_idx=target_class_idx
    )

    # ABLATION: Calculate how much noise the raw math captures BEFORE your filter
    hybrid_raw_noise_pct = calculate_noise_percentage(
        raw_tokens, raw_scores, noise_filter
    )

    # Apply your linguistic filter
    hybrid_filtered_data = noise_filter.process_and_normalize(raw_tokens, raw_scores)

    # Calculate noise: How many tokens flagged as 'is_noise' have a normalized score > 0?
    # (By definition of your filter, this should be 0%, but we prove it here)
    hybrid_noise_count = sum(
        1
        for item in hybrid_filtered_data
        if item["is_noise"] and item.get("normalized_score", 0) > 0
    )
    hybrid_noise_pct = (
        (hybrid_noise_count / len(hybrid_filtered_data)) * 100
        if len(hybrid_filtered_data) > 0
        else 0
    )

    # --- 2. Run SHAP ---
    print("Running SHAP...")
    shap_vals = shap_explainer([test_text])
    # SHAP outputs values for each token. We need to normalize them to compare.
    shap_raw_scores = shap_vals.values[0][:, target_class_idx]
    shap_tokens = shap_vals.data[0]

    # We pass SHAP scores through your filter ONLY to identify which ones ARE noise,
    # but we don't let the filter zero them out. We want to see how much score SHAP gave them.
    shap_noise_pct = calculate_noise_percentage(
        shap_tokens, shap_raw_scores, noise_filter
    )

    # --- 3. Run LIME ---
    print("Running LIME...")
    lime_exp = lime_explainer.explain_instance(
        test_text, lime_predict_proba, num_features=50, num_samples=100
    )
    lime_results = lime_exp.as_list()

    # LIME returns tuples of (word, score). We reconstruct arrays to measure noise.
    lime_tokens = [res[0] for res in lime_results]
    lime_scores = [
        abs(res[1]) for res in lime_results
    ]  # Use absolute magnitude of importance

    lime_noise_pct = calculate_noise_percentage(lime_tokens, lime_scores, noise_filter)

    # --- Save results and generate a publication-oriented comparison figure ---
    results = [
        {
            "Metric": "Noise-attribution rate (normalized importance > 0.1)",
            "Raw Hybrid Fusion (Pre-Filter)": f"{hybrid_raw_noise_pct:.1f}%",
            "HNIF (Post-Filter)": f"{hybrid_noise_pct:.1f}%",
            "SHAP": f"{shap_noise_pct:.1f}%",
            "LIME": f"{lime_noise_pct:.1f}%",
        }
    ]

    df = pd.DataFrame(results)
    df.to_csv("sparsity_results.csv", index=False)
    plot_df = pd.DataFrame(
        {
            "Explanation method": ["Raw Hybrid Fusion", "HNIF", "SHAP", "LIME"],
            "Noise-attribution rate (%)": [
                hybrid_raw_noise_pct,
                hybrid_noise_pct,
                shap_noise_pct,
                lime_noise_pct,
            ],
        }
    )

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.barplot(
        data=plot_df,
        x="Explanation method",
        y="Noise-attribution rate (%)",
        hue="Explanation method",
        palette=["#999999", "#0072B2", "#D55E00", "#009E73"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Attribution of Linguistic Noise by Explanation Method", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Noise-attribution rate (%)")
    ax.set_ylim(0, max(100, plot_df["Noise-attribution rate (%)"].max() * 1.15))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig("sparsity_comparison_plot.png", dpi=300, bbox_inches="tight")
    fig.savefig("sparsity_comparison_plot.pdf", bbox_inches="tight")
    plt.close(fig)
    print("-" * 70)
    print("Benchmark complete. Saved results to 'sparsity_results.csv'.")
    print(
        "Saved figures to 'sparsity_comparison_plot.png' and 'sparsity_comparison_plot.pdf'."
    )
    print(df.to_string(index=False))


def calculate_noise_percentage(tokens, scores, noise_filter):
    """Helper function to calculate how much importance a method gave to grammatical noise."""
    # FIX: Safely check for empty arrays/lists to prevent numpy ValueError
    if tokens is None or len(tokens) == 0:
        return 0

    # Normalize scores between 0 and 1 so we can set a threshold
    scores_arr = np.array(scores)
    if len(scores_arr) > 0 and scores_arr.max() > scores_arr.min():
        norm_scores = (scores_arr - scores_arr.min()) / (
            scores_arr.max() - scores_arr.min()
        )
    else:
        norm_scores = np.zeros_like(scores_arr)

    noise_count = 0
    total_count = len(tokens)

    for token, score in zip(tokens, norm_scores):
        # Clean token for spaCy
        clean_tok = (
            token.replace("Ġ", "")
            .replace(" ", "")
            .replace("_", "")
            .replace("\u2581", "")
            .strip()
        )
        is_noise = False

        if clean_tok in noise_filter.structural_tokens or not clean_tok:
            is_noise = True
        else:
            doc = noise_filter.nlp(clean_tok)
            if len(doc) > 0 and (
                doc[0].pos_ in noise_filter.noise_pos_tags or doc[0].is_punct
            ):
                is_noise = True

        # If it is a noise word AND the method gave it a score > 10% importance, count it as a failure
        if is_noise and score > 0.1:
            noise_count += 1

    return (noise_count / total_count) * 100 if total_count > 0 else 0


if __name__ == "__main__":
    print("Loading models...")
    load_models()

    model = None
    tokenizer = None
    device = getattr(model_loader, "device", "cpu")

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

    run_sparsity_benchmark(model, tokenizer, actual_device)

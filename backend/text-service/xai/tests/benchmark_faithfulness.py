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
    arr = np.array(arr)
    if len(arr) == 0:
        return arr
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def run_faithfulness_benchmark(model, tokenizer, device):
    print("Starting Faithfulness (AOPC) Benchmark")
    print("-" * 70)

    # This mixed-style text provides both conversational and stylometric content.
    test_text = "I think climate change is a big problem today. The multifaceted ramifications of anthropogenic carbon emissions pose existential threats to global ecosystems. We really need to do something about it quickly. Implementing paradigm-shifting legislative frameworks is crucial for systemic remediation."

    target_class_idx = 1  # Assuming 1 is AI-Generated

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

    def predict_proba(texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()

    # --- Extract Hybrid Fusion Scores ---
    raw_scores, raw_tokens = hybrid_extractor.extract_scores(
        test_text, target_class_idx=target_class_idx
    )
    hybrid_filtered = noise_filter.process_and_normalize(raw_tokens, raw_scores)

    words = []
    hybrid_scores_dict = {}
    for item in hybrid_filtered:
        # Ignore structural noise completely
        if item.get("is_noise"):
            continue

        word = item["display_token"].strip()
        if word:
            current_score = hybrid_scores_dict.get(word, 0.0)
            hybrid_scores_dict[word] = max(
                current_score, item.get("normalized_score", 0.0)
            )
            if word not in words:
                words.append(word)

    # --- Extract SHAP Scores ---
    shap_vals = shap_explainer([test_text])
    shap_raw_scores = shap_vals.values[0][:, target_class_idx]
    shap_tokens = shap_vals.data[0]

    shap_scores_dict = {}
    for w in words:
        score = 0.0
        for s_tok, s_val in zip(shap_tokens, shap_raw_scores):
            clean_s = s_tok.replace("Ġ", "").strip()
            if clean_s and clean_s in w:
                score = max(score, float(s_val))
        shap_scores_dict[w] = score

    # --- Extract LIME Scores ---
    lime_exp = lime_explainer.explain_instance(
        test_text, predict_proba, num_features=100, num_samples=100
    )
    lime_results = dict(lime_exp.as_list())

    lime_scores_dict = {}
    for w in words:
        lime_scores_dict[w] = abs(lime_results.get(w.lower(), 0.0))

    # Use the same filtered representation for the baseline and every perturbation.
    # This ensures that the 0% point measures no deletion rather than preprocessing.
    evaluation_text = " ".join(words)

    # --- Rank the words ---
    def get_ranked_words(scores_dict):
        ranked = sorted(words, key=lambda w: scores_dict.get(w, 0.0), reverse=True)
        return ranked

    ranked_hybrid = get_ranked_words(hybrid_scores_dict)
    ranked_shap = get_ranked_words(shap_scores_dict)
    ranked_lime = get_ranked_words(lime_scores_dict)

    print("\nRunning perturbation loop...")

    baseline_conf = float(predict_proba(evaluation_text)[0][target_class_idx])
    print(f"Baseline target-class confidence: {baseline_conf:.4f}")

    mask_percentages = [0, 10, 20, 30, 40, 50]
    results = []

    def mask_text(word_list, ranked_list, pct):
        num_to_mask = int(len(word_list) * (pct / 100))
        words_to_remove = set(ranked_list[:num_to_mask])
        masked = [w for w in word_list if w not in words_to_remove]
        return " ".join(masked)

    for pct in mask_percentages:
        # Hybrid
        masked_hybrid = mask_text(words, ranked_hybrid, pct)
        conf_hybrid = (
            float(predict_proba(masked_hybrid)[0][target_class_idx])
            if masked_hybrid
            else 0.0
        )

        # SHAP
        masked_shap = mask_text(words, ranked_shap, pct)
        conf_shap = (
            float(predict_proba(masked_shap)[0][target_class_idx])
            if masked_shap
            else 0.0
        )

        # LIME
        masked_lime = mask_text(words, ranked_lime, pct)
        conf_lime = (
            float(predict_proba(masked_lime)[0][target_class_idx])
            if masked_lime
            else 0.0
        )

        # Random
        np.random.seed(42)
        ranked_random = list(words)
        np.random.shuffle(ranked_random)
        masked_random = mask_text(words, ranked_random, pct)
        conf_random = (
            float(predict_proba(masked_random)[0][target_class_idx])
            if masked_random
            else 0.0
        )

        # Confidence drop is the faithfulness response at this perturbation level.
        results.append(
            {
                "Masked (%)": pct,
                "HNIF": baseline_conf - conf_hybrid,
                "SHAP": baseline_conf - conf_shap,
                "LIME": baseline_conf - conf_lime,
                "Random Deletion": baseline_conf - conf_random,
            }
        )

        print(
            f"Masked {pct}% -> Drops | HNIF: {(baseline_conf - conf_hybrid):.3f} | SHAP: {(baseline_conf - conf_shap):.3f}"
        )

    df = pd.DataFrame(results)
    aopc_values = {
        method: float(np.trapz(df[method], df["Masked (%)"]) / 100.0)
        for method in ["HNIF", "SHAP", "LIME", "Random Deletion"]
    }
    df.to_csv("faithfulness_aopc_results.csv", index=False)

    print("\nGenerating Faithfulness Plot...")
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        df["Masked (%)"],
        df["HNIF"],
        marker="o",
        label="HNIF",
        linewidth=2.5,
        color="#2563eb",
    )
    ax.plot(
        df["Masked (%)"],
        df["SHAP"],
        marker="s",
        label="SHAP",
        linewidth=2,
        color="#dc2626",
    )
    ax.plot(
        df["Masked (%)"],
        df["LIME"],
        marker="^",
        label="LIME",
        linewidth=2,
        color="#16a34a",
    )
    ax.plot(
        df["Masked (%)"],
        df["Random Deletion"],
        marker="x",
        label="Random Deletion",
        linewidth=2,
        linestyle="--",
        color="gray",
    )

    ax.set_title("Faithfulness Under Progressive Token Deletion", pad=12)
    ax.set_xlabel("Top-ranked tokens deleted (%)")
    ax.set_ylabel("Target-class confidence drop")
    ax.set_xticks(mask_percentages)

    # Force Y-axis to start at 0 so we can see the actual drop
    ax.set_ylim(bottom=min(-0.05, float(df.drop(columns=["Masked (%)"]).min().min()) - 0.02))

    ax.legend(title="Explanation method", frameon=True)
    fig.tight_layout()
    fig.savefig("faithfulness_aopc_plot.png", dpi=300, bbox_inches="tight")
    fig.savefig("faithfulness_aopc_plot.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Benchmark complete. Saved results to 'faithfulness_aopc_results.csv'.")
    print("Saved figures to 'faithfulness_aopc_plot.png' and 'faithfulness_aopc_plot.pdf'.")
    print("AOPC values:")
    for method, value in aopc_values.items():
        print(f"  {method}: {value:.4f}")


if __name__ == "__main__":
    load_models()
    model = None
    tokenizer = None
    for attr_name in dir(model_loader):
        attr_val = getattr(model_loader, attr_name)
        if isinstance(attr_val, torch.nn.Module):
            model = attr_val
        if hasattr(attr_val, "convert_ids_to_tokens"):
            tokenizer = attr_val

    actual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(actual_device)
    run_faithfulness_benchmark(model, tokenizer, actual_device)

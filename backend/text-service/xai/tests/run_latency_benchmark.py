import time
import pandas as pd
import torch
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path so we can import your custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shap
from lime.lime_text import LimeTextExplainer
from transformers import pipeline
from xai.core_extractor import HybridFusionExtractor
from classification.model_loader import load_models
import classification.model_loader as model_loader


def run_latency_benchmark(model, tokenizer, device):
    print("Starting explanation-method latency benchmark")
    print("-" * 70)

    # Prepare sample texts of varying lengths
    sample_texts = [
        "This is a short sentence.",
        "The algorithmic progression of artificial intelligence systems demonstrates a remarkable capacity for generating syntactically uniform prose.",
        "Artificial intelligence is rapidly transforming various sectors, including healthcare, finance, and education. Machine learning algorithms, particularly deep learning models like neural networks, are at the forefront of this revolution, enabling systems to learn complex patterns from vast amounts of data.",
        "The development of large language models has sparked intense debate regarding their potential impact on academic integrity. While these tools offer unprecedented capabilities for summarizing information, translating languages, and even generating original text, they also present significant challenges for educators. The primary concern is that students may use these models to automate the writing process, thereby bypassing the critical thinking and analytical skills that essays are designed to evaluate. As a result, educational institutions are increasingly turning to AI detection software to identify machine-generated submissions. However, the efficacy of these detection tools remains a subject of ongoing research, as the boundary between human and machine writing becomes increasingly blurred.",
        "In the context of natural language processing, transformer architectures have achieved state-of-the-art results across a wide range of tasks. The core innovation of the transformer is the self-attention mechanism, which allows the model to weigh the significance of different words in a sequence regardless of their positional distance. This is a significant departure from previous recurrent neural networks (RNNs) and long short-term memory (LSTM) networks, which process data sequentially and often struggle with long-range dependencies. By processing the entire sequence in parallel, transformers not only capture complex contextual relationships more effectively but also benefit from massive parallelization during training on modern GPU hardware. The introduction of models like BERT (Bidirectional Encoder Representations from Transformers) further advanced the field by enabling deep bidirectional pre-training. Unlike traditional unidirectional models, BERT learns representations by conditioning on both left and right context in all layers. This deep contextual understanding is crucial for tasks requiring nuanced comprehension, such as sentiment analysis, question answering, and named entity recognition. As these models continue to scale in parameter count, the need for robust explainability methods becomes increasingly urgent, as their internal decision-making processes grow ever more opaque.",
    ]

    results = []
    target_class_idx = 1  # Assuming 1 is AI-Generated

    # 1. Initialize Hybrid Fusion Extractor
    hybrid_extractor = HybridFusionExtractor(
        model=model, tokenizer=tokenizer, ig_steps=10
    )

    # 2. Initialize SHAP Explainer
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1,
    )
    shap_explainer = shap.Explainer(pipe)

    # 3. Initialize LIME Explainer & Custom Predictor
    lime_explainer = LimeTextExplainer(class_names=["Human", "AI-Generated"])

    def lime_predict_proba(texts):
        """LIME requires a function that takes raw strings and returns numpy probabilities."""
        # Process in chunks if LIME sends too many texts at once
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()

    for i, text in enumerate(sample_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        num_tokens = inputs["input_ids"].shape[1]
        print(f"Testing Sequence {i+1}: {num_tokens} tokens...")

        # --- Benchmark 1: Hybrid Fusion ---
        start_time = time.time()
        hybrid_extractor.extract_scores(text, target_class_idx=target_class_idx)
        hybrid_time = (time.time() - start_time) * 1000

        # --- Benchmark 2: SHAP ---
        start_time = time.time()
        _ = shap_explainer([text])
        shap_time = (time.time() - start_time) * 1000

        # --- Benchmark 3: LIME ---
        start_time = time.time()
        # Note: We restrict LIME to 50 samples to prevent Colab from timing out completely.
        # Even with just 50 samples, it will demonstrate severe latency.
        _ = lime_explainer.explain_instance(
            text, lime_predict_proba, num_features=50, num_samples=50
        )
        lime_time = (time.time() - start_time) * 1000

        # Save results
        results.append(
            {
                "num_tokens": num_tokens,
                "hybrid_time_ms": hybrid_time,
                "shap_time_ms": shap_time,
                "lime_time_ms": lime_time,
            }
        )

        print(
            f"  -> Hybrid: {hybrid_time:.2f}ms | SHAP: {shap_time:.2f}ms | LIME: {lime_time:.2f}ms"
        )

    # Save tabular results and a publication-oriented latency figure.
    df = pd.DataFrame(results)
    df.to_csv("latency_results_full.csv", index=False)
    plot_df = df.melt(
        id_vars="num_tokens",
        value_vars=["hybrid_time_ms", "shap_time_ms", "lime_time_ms"],
        var_name="method",
        value_name="latency_ms",
    )
    plot_df["method"] = plot_df["method"].map(
        {
            "hybrid_time_ms": "HNIF",
            "shap_time_ms": "SHAP",
            "lime_time_ms": "LIME",
        }
    )

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.lineplot(
        data=plot_df,
        x="num_tokens",
        y="latency_ms",
        hue="method",
        style="method",
        markers=True,
        dashes=False,
        palette={"HNIF": "#0072B2", "SHAP": "#D55E00", "LIME": "#009E73"},
        linewidth=1.8,
        markersize=6,
        ax=ax,
    )
    ax.set_title("Explanation Latency as a Function of Input Length", pad=12)
    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel("Runtime (ms, logarithmic scale)")
    ax.set_yscale("log")
    ax.legend(title="Explanation method", frameon=True)
    fig.tight_layout()
    fig.savefig("latency_comparison_plot.png", dpi=300, bbox_inches="tight")
    fig.savefig("latency_comparison_plot.pdf", bbox_inches="tight")
    plt.close(fig)
    print("-" * 70)
    print("Benchmark complete. Saved results to 'latency_results_full.csv'.")
    print(
        "Saved figures to 'latency_comparison_plot.png' and 'latency_comparison_plot.pdf'."
    )


if __name__ == "__main__":
    print("Loading models (this takes a few seconds)...")
    load_models()

    model = None
    tokenizer = None

    # Grab the model and tokenizer from Ishara's loader
    for attr_name in dir(model_loader):
        attr_val = getattr(model_loader, attr_name)
        if isinstance(attr_val, torch.nn.Module):
            model = attr_val
            print(f"✅ Found PyTorch Model in variable: {attr_name}")
        if hasattr(attr_val, "convert_ids_to_tokens"):
            tokenizer = attr_val
            print(f"✅ Found Tokenizer in variable: {attr_name}")

    if model is None or tokenizer is None:
        print(
            "❌ Error: Could not automatically detect the DeBERTa model or tokenizer in model_loader.py."
        )
        sys.exit(1)

    # FORCE THE MODEL TO THE GPU
    print("Forcing model to GPU to ensure benchmark accuracy...")
    actual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(actual_device)

    run_latency_benchmark(model, tokenizer, actual_device)

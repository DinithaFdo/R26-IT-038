# Robust, Explainable Multi-Modal Classification of Deepfake Voice and AI-Generated Text

Project ID: R26-038
Institution: Sri Lanka Institute of Information Technology (SLIIT)
Department: Information Technology
Degree: B.Sc. (Hons) in Information Technology
Submission: March 2026


## Table of Contents

1. Project Overview
2. Research Motivation
3. System Architecture
4. Components
   - Component 1 — Multi-Branch Deepfake Voice Detection
   - Component 2 — Explainable AI Text Classification (HNIF)
   - Component 3 — Data-Centric AI Detection Engine
   - Component 4 — Explainable Spoof Voice Analysis System (ESVAS)
5. Research Gaps Addressed
6. Datasets
7. Technology Stack
8. Evaluation Metrics
9. Project Timeline
10. Repository Structure
11. Setup and Installation
12. Team
13. Supervisors
14. SDG Alignment
15. References
16. License


## Project Overview
This research project develops a multi-modal, explainable classification framework capable of detecting AI-generated synthetic speech (deepfake voice) and AI-generated text. The system is designed to address two interrelated problems in the AI security domain:

- **Deepfake Voice Detection** — Identifying synthetic or manipulated speech produced by modern text-to-speech (TTS) and voice conversion systems.
- **AI-Generated Text Detection** — Classifying text produced by Large Language Models (LLMs) with transparent, mathematically validated explanations.

The framework is built around four tightly integrated components developed by individual researchers, each targeting a specific architectural or analytical gap in the state of the art.

The broader system delivers:

- High-accuracy classification of both synthetic audio and synthetic text
- Mathematically validated, human-readable explanations for every decision
- Robustness against adversarial evasion attacks
- Fairness towards non-native English writers
- A modular, scalable architecture suitable for deployment in banking, cybersecurity, and digital identity platforms


## Research Motivation

### The Problem
Modern AI tools can clone a human voice from just a few seconds of audio and generate synthetic speech that is indistinguishable to an untrained listener. Similarly, LLMs like ChatGPT produce text that is increasingly difficult to distinguish from human writing.

These capabilities are actively exploited for:

- Telephone fraud and impersonation attacks
- Corporate executive voice cloning for unauthorized financial transactions
- Academic integrity violations using AI-generated submissions
- Social engineering campaigns using synthetic audio and text

### Scale of Impact

| Domain | Statistic |
|--------|-----------|
| Cybercrime losses (FBI IC3, 2024) | $16.6 billion |
| Adults who experienced AI voice scams (McAfee) | 1 in 4 |
| Projected U.S. AI-driven fraud losses by 2027 (Deloitte) | $40 billion |
| Commercial detectors falsely flagging ESL essays as AI (Stanford, 2023) | 61% |
| Drop in detector accuracy after paraphrasing attacks | >95% → <40% |

### Why Existing Systems Fall Short
Current commercial detectors (GPTZero, Turnitin, Reality Defender, Pindrop) share three fundamental limitations:

1. **Opacity** — They output binary verdicts or probability scores with no explanation of the reasoning
2. **Brittleness** — They fail when subjected to adversarial paraphrasing, synonym swapping, or voice conversion techniques
3. **Bias** — They systematically misclassify formal, low-burstiness human writing (common in ESL writers) as AI-generated

This project directly addresses all three limitations through a unified, explainable, multi-modal detection architecture.


## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                      │
│              Audio (.wav)  │  Text (essays / documents)                 │
└───────────────┬────────────────────────────────┬────────────────────────┘
                │                                │
                ▼                                ▼
┌──────────────────────────┐    ┌────────────────────────────────────────┐
│  COMPONENT 1             │    │  COMPONENT 3                           │
│  Multi-Branch Voice      │    │  Data-Centric AI Text Detection        │
│  Detection System        │    │  (Sanitization + DistilBERT + XGBoost) │
│  (Wijesundara)           │    │  (Athapaththu)                         │
└──────────┬───────────────┘    └───────────────────┬────────────────────┘
           │                                        │
           │  Intermediate Representations          │  Branch Scores
           │  (Attention Tensors, Feature Vectors)  │  + Feature Arrays
           ▼                                        ▼
┌──────────────────────────┐    ┌────────────────────────────────────────┐
│  COMPONENT 4             │    │  COMPONENT 2                           │
│  Explainable Voice       │    │  Hybrid Neural Interpretability        │
│  Analysis System (ESVAS) │    │  Framework (HNIF) + Agentic SDK        │
│  (Silva)                 │    │  (Fernando)                            │
└──────────┬───────────────┘    └───────────────────┬────────────────────┘
           │                                        │
           ▼                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     UNIFIED OUTPUT DASHBOARD                            │
│   Classification Label │ Confidence Score │ Explanation Report         │
│   Temporal Heatmap     │ Feature Attribution │ Faithfulness Score       │
└─────────────────────────────────────────────────────────────────────────┘
```


## Components

### Component 1 — Multi-Branch Deepfake Voice Detection
**Researcher:** Aweesha Thavishanka Wijesundara
**Focus:** Robust, high-accuracy detection of synthetic and manipulated speech

#### Overview
This component introduces a Multi-Branch Deepfake Voice Detection System that processes audio through four parallel, specialized detection branches. Each branch captures a distinct type of spoofing evidence. Their outputs are combined via an adaptive score-level fusion mechanism.

#### Detection Branches

| Branch | Method | Target Artifact |
|--------|--------|----------------|
| Branch 1 — Spectral | LFCC (180-dim) + CNN / TCN | Low-level spectral artifacts from vocoders and TTS |
| Branch 2 — Spectro-Temporal | AASIST (Graph Attention Network) | Heterogeneous spectro-temporal inconsistencies |
| Branch 3 — Temporal Coherence | WavLM / XLSR + BiLSTM / Mamba | Long-range temporal inconsistencies in synthetic speech |
| Branch 4 — Physiological | IAIF + NAQ, GNE, H1–H2, QOQ + MLP / RF | Biological glottal excitation patterns |

#### Adaptive Fusion
The final classification score is computed as a weighted combination of all four branch scores:
```
Final Score = w₁S₁ + w₂S₂ + w₃S₃ + w₄S₄
              where Σwᵢ = 1 (weights learned via attention-based mechanism)
```

#### Key Design Decisions
- Each branch is developed and validated independently before integration
- Modular architecture allows replacement of individual branches without full retraining
- Glottal source analysis is included as a dedicated physiological evidence stream — a novel contribution rarely found in benchmark systems
- Adaptive weight learning replaces static weighted averaging used in prior fusion approaches

#### Evaluation Datasets

| Dataset | Purpose |
|---------|---------|
| ASVspoof 2019 LA | Primary training and evaluation |
| ASVspoof 2021 LA | Cross-dataset robustness testing |
| PartialSpoof | Temporal localization evaluation |
| In-the-Wild | Generalization to real-world conditions |

---

### Component 2 — Explainable AI Text Classification (HNIF)
**Researcher:** Fernando M.D.T (IT22090744)
**Focus:** Mathematically validated, human-readable explanations for AI text classification decisions

#### Overview
This component introduces the Hybrid Neural Interpretability Framework (HNIF) integrated with a Large Language Model-driven Agentic SDK. It operates as a non-invasive analytical layer on top of the DeBERTa-v3 text classifier, extracting and mathematically filtering internal model signals to produce stable, faithful, human-readable explanation reports.

#### Core Innovation — Hybrid Saliency Fusion
The core novelty is the element-wise multiplication of two complementary internal signals:
```
Hybrid Saliency = Normalize(Attention Rollout) × Normalize(Integrated Gradients)
```

The multiplication acts as a noise-canceling filter retaining only tokens that are both structurally prominent and causally significant.

#### Processing Pipeline
```
DeBERTa-v3 Inference
        │
        ▼
[1] Extraction Engine — PyTorch hooks → Attention Rollout + Integrated Gradients
[2] Hybrid Fusion Engine — Normalize(Attention) × Normalize(Gradients)
[3] Linguistic Smoothing — Sub-word aggregation + POS masking
[4] Faithfulness Validation — AOPC score
[5] Agentic Translation — LLM API → natural language report → Streamlit dashboard
```

#### Specific Objectives

| Sub-Objective | Description |
|--------------|-------------|
| SO1 | Implement PyTorch hooks to extract attention tensors and gradient attributions |
| SO2 | Develop the hybrid saliency fusion (Attention Rollout × Integrated Gradients) |
| SO3 | Sub-word aggregation and POS masking for linguistic coherence |
| SO4 | Automated AOPC faithfulness validation loop |
| SO5 | Agentic SDK for natural language explanation generation and interactive dashboard |

#### Tools

| Tool | Role |
|------|------|
| PyTorch | Dynamic computation graph, backward hooks |
| Captum (Meta) | Optimized Integrated Gradients implementation |
| spaCy | POS tagging, sub-word aggregation |
| Gemini / OpenAI API | LLM-driven explanation generation |
| Streamlit | Interactive explanation dashboard |

---

### Component 3 — Data-Centric AI Detection Engine
**Researcher:** A.W.M. Ishara Udayanga Athapaththu (IT22251046)
**Focus:** Adversarially robust AI-generated text detection using multi-branch linguistic analysis

#### Overview
This component addresses two fundamental failures in commercial AI text detectors:

1. **Adversarial Vulnerability** — Detectors are bypassed by invisible Unicode characters, Cyrillic homoglyphs, and structural noise. Demonstrated experimentally: a homoglyph attack reduced GPTZero confidence from 100% to 89%.
2. **Single-Feature Limitation** — Existing detectors rely solely on perplexity and burstiness, missing deeper semantic and lexical patterns that characterise AI writing.

The solution is a layered detection pipeline: an adversarial sanitization layer followed by two independently validated detection branches, with multi-branch fusion and further branches planned as future work.

#### Detection Branches

| Branch | Method | Accuracy | Status |
|--------|--------|----------|--------|
| Branch 1 — Semantic | Fine-tuned DistilBERT | 95.35% / AUROC 99.16% | ✅ Complete |
| Branch 2 — Lexical | XGBoost + 11 Stylometric Features | 80.84% / AUROC 89.32% | ✅ Complete |
| Branch 3 — Syntactic | POS-based analysis | — | ❌ Excluded (research finding) |
| Branch 4 — N-gram | TF-IDF statistical model | 88.51% | ⏳ Trained, pending fusion |

Branch 1 was fine-tuned on a combined dataset of 19,144 samples merging the RAID benchmark and ai-text-detection-pile across 6 domains and 7 AI models. Branch 2 extracts 11 handcrafted stylometric features including Type-Token Ratio, pronoun ratio, sentence length variance, and contraction ratio using NLTK and spaCy. Branch 3 was evaluated and excluded due to noise sensitivity — documented as a research finding. Branch 4 is trained and awaiting integration.

#### Input Sanitization Layer ✅ Complete

Implemented as a standalone Python module using deterministic regex and Unicode normalization — no external API, zero latency, content-safe. Handles 10 adversarial attack categories including zero-width characters, Cyrillic homoglyph substitution, mathematical Unicode, null bytes, HTML entities, URLs, Markdown patterns, repeated punctuation, and whitespace manipulation. Verified by 9 automated unit tests (9/9 PASS) and confirmed 100% detection across 10 real attack test cases.

#### Future Work

Multi-branch weighted fusion integrating all trained branches, confidence-aware structured JSON output for Component 2 (XAI), domain-aware multi-task DistilBERT branch, and a stylistic inconsistency branch to reduce false positives on ESL writing are planned for the next project phase.

#### Sub-Objectives Status

| # | Sub-Objective | Status |
|---|--------------|--------|
| 1 | Implement input sanitization layer neutralizing adversarial evasion attacks | ✅ Complete |
| 2 | Develop semantic detection branch using fine-tuned DistilBERT on domain-diverse dataset | ✅ Complete |
| 3 | Develop interpretable lexical analysis branch using stylometric features and XGBoost | ✅ Complete |
| 4 | Integrate branches through accuracy-weighted fusion with confidence-aware JSON output | ⏳ Future Work |
| 5 | Implement domain-aware multi-task detection branch for cross-domain generalisation | ⏳ Future Work |
| 6 | Implement stylistic inconsistency branch to reduce false positives on ESL writing | ⏳ Future Work |

#### Evaluation Datasets

| Dataset | Description | Usage |
|---------|-------------|-------|
| RAID Benchmark (Dugan et al., ACL 2024) | 6M+ samples, 11 models, 8 domains, 11 attack types | Primary training source |
| ai-text-detection-pile (HuggingFace) | Multi-domain AI and human text | Diversity augmentation |
| Combined Dataset (constructed) | 19,144 balanced samples across 6 domains | Training, validation, and test set |

---

### Component 4 — Explainable Spoof Voice Analysis System (ESVAS)
**Researcher:** Silva S.P.S (IT22219602)
**Focus:** Temporal and semantic explainability for deepfake voice classification decisions

#### Overview
This component introduces the Explainable Spoof Voice Analysis System (ESVAS), a non-invasive analytical layer that receives intermediate representations from the multi-branch voice classifier (Component 1) and generates a structured, dual-view explanation report combining:

- **Temporal localization** — Where in the audio does the anomaly occur?
- **Acoustic feature attribution** — What physical property indicates synthetic origin?

#### Four-Phase Pipeline
**Phase 1 — Feature and Attention Extraction**
PyTorch `register_forward_hook()` intercepts outputs from all four detection branches during inference. Extracted features include 19 LFCC coefficients, F0 mean and variance, jitter, shimmer, HNR, and GCI irregularity via REAPER. All tensors stored in asynchronous in-memory buffers.

**Phase 2 — Temporal Attention Visualization via Attention Rollout**
```
R_l = Â_l × R_{l-1}    where    Â_l = 0.5 × A_l + 0.5 × I
```
Produces an Attention Rollout Spectrogram (ARS) mapped per 20ms audio frame. Regions above the 80th percentile are marked as primary anomaly candidates. Novel contribution: first application of Attention Rollout to XLSR for voice anti-spoofing.

**Phase 3 — Feature-Level Semantic Explanation via XGBoost-SHAP**
An XGBoost surrogate model trained on physical acoustic features. SHAP decomposes each prediction into signed feature attributions (positive → Spoof, negative → Bonafide) with outputs including ranked importance charts, beeswarm plots, and per-sample waterfall plots.

**Phase 4 — Unified Synthesis and Faithfulness Evaluation**
```
IoU(ARS high-attention regions, SHAP top-3 feature time windows) > 0.75 = Consistent ✓
AOPC = (1 / (L+1)) × Σ [ f(x) − f(x \ Ω_l^k) ]
```

#### Key Design Properties
- Non-invasive — PyTorch hooks attach without modifying classifier logic or weights
- Unidirectional data flow — Explanation layer cannot influence classifier predictions
- Modular — Each phase is independently replaceable
- Audit-ready output — Structured dashboard with natural-language summary for non-specialist users

#### Evaluation Datasets

| Dataset | Size | Role |
|---------|------|------|
| ASVspoof 2019 LA | ~121K utterances | XGBoost surrogate training and validation |
| ASVspoof 2021 LA | ~180K utterances | Temporal localization evaluation (with channel effects) |
| FakeSound2 | ~30K clips (segment annotations) | Ground-truth IoU measurement and AOPC validation |


## Research Gaps Addressed

| Gap | Description | Addressed By |
|-----|-------------|-------------|
| G1 — Limited architectural diversity | Most systems compress all spoof evidence into a single embedding | Component 1 (4 independent branches) |
| G2 — Absence of physiological modeling | Glottal source features ignored by deep learning era systems | Component 1 (Branch 4 — Glottal Analysis) |
| G3 — No temporal localization | Systems classify utterances without identifying manipulated segments | Component 4 (Attention Rollout → ARS) |
| G4 — Suboptimal fusion mechanisms | Simple weighted averaging fails to optimize complementary signals | Component 1 (Adaptive learned fusion) |
| G5 — Black-box opacity | No explanation of why a classification was made | Components 2 and 4 |
| G6 — No faithfulness validation | Explanation heatmaps lack mathematical proof of causal accuracy | Components 2 and 4 (AOPC metric) |
| G7 — Single-feature text detection | Existing detectors rely solely on perplexity and burstiness | Component 3 (Branch 1 + Branch 2) |
| G8 — Adversarial text evasion vulnerability | Detection degrades under Unicode manipulation and hidden character injection | Component 3 (Input sanitization layer) |
| G9 — No multi-branch text fusion | No existing system combines semantic and lexical signals in a unified framework | Component 3 (Planned — future work) |
| G10 — Uninterpretable XAI outputs | SHAP on spectrogram pixels is not acoustically meaningful | Component 4 (SHAP on physical acoustic features) |


## Datasets

| Dataset | Domain | Size | Access |
|---------|--------|------|--------|
| ASVspoof 2019 LA | Deepfake voice | ~121K utterances | Public (research license) |
| ASVspoof 2021 LA | Deepfake voice (telephony conditions) | ~180K utterances | Public (research license) |
| PartialSpoof | Partially manipulated audio | Annotated segments | Public (research license) |
| In-the-Wild | Real-world synthetic voice | Variable | Public |
| FakeSound2 | Deepfake audio (with segment timestamps) | ~30K clips | Public |
| RAID Benchmark | Adversarially modified AI text | 6M+ samples | Public |
| ai-text-detection-pile | Multi-domain AI and human text | Large-scale | Public (HuggingFace) |
| Combined Text Dataset (constructed) | Merged AI and human text across 6 domains | 19,144 samples | Constructed |

> **Ethical Note:** All datasets are publicly licensed for academic research. No private voice or text data is collected. No personally identifiable information (PII) is processed or stored. All systems are designed exclusively for detection — no synthesis or impersonation functionality is implemented.


## Technology Stack

### Core ML & Audio

| Library | Version | Usage |
|---------|---------|-------|
| Python | 3.10+ | Primary development language |
| PyTorch | 2.x | Deep learning framework, forward/backward hooks |
| HuggingFace Transformers | Latest | XLSR-53, WavLM, DistilBERT model access |
| Librosa | 0.10 | Audio feature extraction, spectrogram generation |
| Parselmouth (Praat API) | Latest | Jitter, shimmer, HNR extraction |
| XGBoost | 2.0 | Lexical branch classifier and surrogate model for SHAP |
| SHAP | 0.44 | Shapley value computation (TreeExplainer) |
| Captum (Meta) | Latest | Optimized Integrated Gradients |
| scikit-learn | Latest | StandardScaler, evaluation metrics |
| NLTK | Latest | Tokenization, lexical feature extraction |
| spaCy | Latest | POS tagging, sentence segmentation |

### NLP & Detection

| Library | Version | Usage |
|---------|---------|-------|
| HuggingFace Transformers | Latest | DistilBERT fine-tuning and inference |
| NLTK | Latest | Stylometric feature extraction — Branch 2 |
| spaCy | Latest | Sentence parsing, POS tagging |
| re / unicodedata | Built-in | Regex-based sanitization and NFKC normalization |

### Signal Processing

| Library | Usage |
|---------|-------|
| IAIF Toolkit | Glottal source feature extraction (NAQ, GNE, QOQ) |
| REAPER | Glottal Closure Interval (GCI) estimation |
| Librosa | LFCC, mel-spectrogram, F0 (PYIN) |

### Infrastructure & UI

| Tool | Usage |
|------|-------|
| FastAPI | REST API for model serving |
| Streamlit | Interactive explanation dashboard |
| React.js | Frontend user interface |
| D3.js / Plotly | Feature visualization |
| Docker | Containerized deployment |
| Weights & Biases | Experiment tracking |
| Google Colab Pro / AWS EC2 | Cloud GPU training (NVIDIA A100) |


## Evaluation Metrics

### Voice Detection (Components 1 & 4)

| Metric | Description |
|--------|-------------|
| Equal Error Rate (EER) | Primary benchmark metric — point where FAR equals FRR |
| Accuracy | Overall correct classifications |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under the receiver operating characteristic curve |
| Cross-dataset EER | Robustness to unseen datasets and attack types |
| Ablation Study | Performance impact of removing individual branches |

### Text Detection (Component 3)

| Metric | Description | Result |
|--------|-------------|--------|
| Accuracy | Overall classification correctness | Branch 1: 95.35% / Branch 2: 80.84% |
| F1 Score | Balanced precision and recall | Branch 1: 95.35% / Branch 2: 79.87% |
| AUROC | Discrimination ability | Branch 1: 99.16% / Branch 2: 89.32% |
| Real-World Validation | Performance on unseen texts | Branch 1: 6/6 / Branch 2: 6/8 |
| Sanitization Detection Rate | Attack coverage | 10/10 — 100% |
| Unit Tests | Sanitization correctness | 9/9 PASS |

### Explainability Validation (Components 2 & 4)

| Metric | Description |
|--------|-------------|
| AOPC | Area Over Perturbation Curve — faithfulness of highlighted tokens/features |
| IoU | Intersection-over-Union between temporal attention regions and SHAP feature windows |
| Surrogate Fidelity (R²) | How accurately XGBoost mirrors the black-box classifier (target: R² > 0.90) |
| Explanation Consistency | Identical inputs must produce identical explanations (deterministic) |
| Human Agreement (HA) | Alignment between explanation highlights and human-readable acoustic markers |


## Project Timeline

| Phase | Period | Duration | Milestone |
|-------|--------|----------|-----------|
| Ideation & Planning | Nov 2025 – Dec 2025 | 8 weeks | Research idea finalized |
| Formalization & Proposal | Jan 2026 – Mar 2026 | 10 weeks | Charter submitted; Proposal defense |
| Data & Feature Development | Apr 2026 – May 2026 | ~7 weeks | Preprocessing pipelines complete |
| Model Development | May 2026 – Jun 2026 | 6 weeks | All detection branches trained |
| Integration & Testing | Jun 2026 – Jul 2026 | 6 weeks | Prototype completed |
| Evaluation & Optimization | Aug 2026 – Sep 2026 | 8 weeks | Final system validated |
| Documentation & Submission | Oct 2026 | 2 weeks | Final research submission |


## Repository Structure

```
R26-038/
│
├── README.md
│
├── component_1_voice_detection/          # Multi-Branch Deepfake Voice Detection
│   ├── branches/
│   │   ├── branch1_spectral/
│   │   ├── branch2_aasist/
│   │   ├── branch3_temporal/
│   │   └── branch4_glottal/
│   ├── fusion/
│   │   └── adaptive_fusion.py
│   ├── preprocessing/
│   │   └── audio_preprocessor.py
│   ├── train.py
│   ├── evaluate.py
│   └── requirements.txt
│
├── component_2_text_xai/                 # Hybrid Neural Interpretability Framework
│   ├── extraction/
│   │   └── hook_engine.py
│   ├── fusion/
│   │   └── hybrid_saliency_fusion.py
│   ├── smoothing/
│   │   └── linguistic_smoother.py
│   ├── validation/
│   │   └── aopc_validator.py
│   ├── agentic_sdk/
│   │   └── llm_agent.py
│   ├── dashboard/
│   │   └── app.py
│   └── requirements.txt
│
├── component_3_text_detection/           # Data-Centric AI Detection Engine
│   ├── sanitization/
│   │   ├── sanitization.py              # Sanitization layer — 10 attack types ✅
│   │   └── sanitization_test.ipynb      # Panel demonstration notebook ✅
│   ├── branch1_semantic/
│   │   ├── train_distilbert.py          # DistilBERT fine-tuning pipeline ✅
│   │   └── branch1_semantic_model/      # Saved model checkpoint ✅
│   ├── branch2_lexical/
│   │   ├── feature_extractor.py         # 11 stylometric feature extraction ✅
│   │   ├── train_xgboost.py             # XGBoost training pipeline ✅
│   │   └── branch2_lexical_model_v3/    # Saved model + scaler + feature names ✅
│   ├── branch4_ngram/
│   │   └── branch4_ngram_model/         # Trained — pending fusion integration ⏳
│   ├── fusion/
│   │   └── fusion_engine.py             # Future work ⏳
│   ├── data/
│   │   ├── combined_dataset.csv         # 19,144 balanced samples ✅
│   │   └── raid_loader.py               # Streaming dataset loader ✅
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── visualizations/
│   │   ├── branch1_training_curves.png
│   │   ├── branch1_confusion_matrix.png
│   │   └── branch2_feature_importance.png
│   └── requirements.txt
│
├── component_4_voice_xai/               # Explainable Spoof Voice Analysis System
│   ├── extraction/
│   │   └── hook_interface.py
│   ├── temporal_xai/
│   │   └── attention_rollout.py
│   ├── semantic_xai/
│   │   ├── surrogate_model.py
│   │   └── shap_explainer.py
│   ├── evaluation/
│   │   ├── aopc_evaluator.py
│   │   └── iou_consistency.py
│   ├── dashboard/
│   │   └── esvas_dashboard.py
│   └── requirements.txt
│
├── shared/
│   ├── datasets/
│   ├── configs/
│   └── utils/
│
├── docs/
│   ├── proposal_reports/
│   ├── architecture_diagrams/
│   └── evaluation_results/
│
└── notebooks/
    ├── eda_audio.ipynb
    ├── eda_text.ipynb
    └── ablation_study.ipynb
```


## Setup and Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (NVIDIA, minimum 16GB VRAM recommended)
- Git

### Clone the Repository
```bash
git clone https://github.com/SLIIT-R26-038/deepfake-multimodal-detection.git
cd deepfake-multimodal-detection
```

### Component 1 — Voice Detection
```bash
cd component_1_voice_detection
pip install -r requirements.txt
python train.py --config configs/full_pipeline.yaml
python evaluate.py --dataset asvspoof2021 --checkpoint checkpoints/best_model.pt
```

### Component 2 — Text Explainability (HNIF)
```bash
cd component_2_text_xai
pip install -r requirements.txt
streamlit run dashboard/app.py
python validation/aopc_validator.py --model deberta-v3 --dataset raid
```

### Component 3 — AI Text Detection
```bash
cd component_3_text_detection
pip install -r requirements.txt

# Verify sanitization layer (runs 9 built-in unit tests)
python sanitization/sanitization.py

# Train Branch 1 — DistilBERT (GPU recommended)
python branch1_semantic/train_distilbert.py \
    --dataset data/combined_dataset.csv \
    --epochs 3 \
    --output branch1_semantic/branch1_semantic_model

# Train Branch 2 — XGBoost (CPU sufficient)
python branch2_lexical/train_xgboost.py \
    --dataset data/combined_dataset.csv \
    --output branch2_lexical/branch2_lexical_model_v3

# Evaluate branches
python evaluation/evaluate.py \
    --branch1_model branch1_semantic/branch1_semantic_model \
    --branch2_model branch2_lexical/branch2_lexical_model_v3
```

### Component 4 — Voice Explainability (ESVAS)
```bash
cd component_4_voice_xai
pip install -r requirements.txt
streamlit run dashboard/esvas_dashboard.py
python evaluation/aopc_evaluator.py --audio samples/test.wav
python evaluation/iou_consistency.py --dataset fakesound2
```

### Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token
```


## Team

| Member | ID | Component | Research Focus |
|--------|----|-----------|----------------|
| Aweesha Thavishanka Wijesundara | IT22183668 | Component 1 | Multi-Branch Deepfake Voice Detection |
| Fernando M.D.T | IT22090744 | Component 2 | Hybrid Neural Interpretability Framework (HNIF) |
| A.W.M. Ishara Udayanga Athapaththu | IT22251046 | Component 3 | Data-Centric AI Text Detection Engine |
| Silva S.P.S | IT22219602 | Component 4 | Explainable Spoof Voice Analysis System (ESVAS) |


## Supervisors

| Role | Name |
|------|------|
| Internal Supervisor | Dr. Kalpani Manathunga |
| Co-Supervisor | Ms. Poojani Gunathilake |
| External Supervisor | Mr. Samitha Vidhanaarachchi |
| Department | Information Technology, SLIIT |


## SDG Alignment

| SDG | Relevance |
|-----|-----------|
| SDG 4 — Quality Education | Reduces algorithmic bias against ESL students in AI detection systems; protects academic integrity fairly |
| SDG 9 — Industry, Innovation, and Infrastructure | Advances transparent, trustworthy AI infrastructure for secure digital systems |
| SDG 10 — Reduced Inequalities | Mitigates bias against non-native English speakers and writers from developing nations |
| SDG 16 — Peace, Justice, and Strong Institutions | Supports institutional integrity by detecting digital impersonation and synthetic media fraud |


## References
Selected key references across all components:

1. Jung et al., "AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks," ICASSP, 2022.
2. Tak et al., "End-to-End Anti-Spoofing with RawNet2," ICASSP, 2021.
3. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS, 2020.
4. Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," IEEE JSTSP, 2022.
5. Sanh et al., "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv, 2019.
6. Sundararajan et al., "Axiomatic Attribution for Deep Networks," ICML, 2017.
7. Abnar & Zuidema, "Quantifying Attention Flow in Transformers," ACL, 2020.
8. Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions (SHAP)," NeurIPS, 2017.
9. Samek et al., "Evaluating the Visualization of What a Deep Neural Network Has Learned (AOPC)," IEEE TNNLS, 2017.
10. Liang et al., "GPT Detectors are Biased Against Non-Native English Writers," Patterns, 2023.
11. Dugan et al., "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors," ACL, 2024.
12. Creo & Pudasaini, "SilverSpeak: Evading AI-Generated Text Detectors Using Homoglyphs," ACL GenAIDetect, 2025.
13. Alku, "Glottal Wave Analysis with Pitch Synchronous Iterative Adaptive Inverse Filtering," Speech Communication, 1992.
14. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR, 2023.
15. Zhao et al., "FakeSound2: A Benchmark for Explainable and Generalizable Deepfake Sound Detection," arXiv, 2025.
16. Mersha et al., "Evaluating the Effectiveness of XAI Techniques for Encoder-Based Language Models," arXiv, 2025.

> Full reference lists are available in each component's individual proposal report under `/docs/proposal_reports/`.


## License
This project is developed for academic research purposes at the Sri Lanka Institute of Information Technology (SLIIT). All datasets used are publicly licensed for academic research. The system is designed exclusively as a defensive detection tool. No voice synthesis, text generation, or impersonation functionality is implemented or distributed.

---

<p align="center">
  <strong>Sri Lanka Institute of Information Technology (SLIIT)</strong><br>
  Department of Information Technology · R26-038 · March 2026
</p>

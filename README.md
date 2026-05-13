# Robust, Explainable Multi-Modal Classification of Deepfake Voice and AI-Generated Text (R26-038)

## 📌 Project Overview
The rapid advancement of Generative Artificial Intelligence (GenAI) has democratized the creation of hyper-realistic synthetic media, introducing severe vulnerabilities across digital platforms, educational institutions, and financial sectors. Existing detection tools for AI-generated text and deepfake voice often act as opaque "black-box" systems—providing arbitrary probability scores without verifiable, mathematically grounded reasoning. Furthermore, these commercial systems suffer from critical flaws: severe vulnerability to evasion attacks (like paraphrasing) and documented algorithmic bias against non-native speakers.

This research project introduces a comprehensive, multi-modal detection and interpretability framework. By combining a multi-branch deepfake voice classifier and a data-centric AI text detector with advanced Explainable AI (XAI) pipelines, the system not only accurately identifies synthetic media but also provides transparent, mathematically validated, and natural-language explanations for its decisions. 

## ✨ Key Features
* **Multi-Branch Voice Classification:** Evaluates audio through spectral, spectro-temporal, long-range temporal, and glottal source branches with an adaptive score-level fusion mechanism for maximum robustness.
* **Fair & Robust Text Detection:** Utilizes active agentic preprocessing and DeBERTa-v3 to resist paraphrasing attacks while eliminating algorithmic bias against English as a Second Language (ESL) writers.
* **Explainable Voice Analysis:** Maps transformer attention directly to audio spectrograms (Attention Rollout) and provides semantic feature attribution via XGBoost and SHAP.
* **Hybrid Neural Interpretability for Text:** Mathematically fuses Attention Rollout and Integrated Gradients to act as a noise-canceling filter, translating abstract mathematical weights into human-readable audit reports.
* **Quantitative Faithfulness Validation:** Uses the Area Over the Perturbation Curve (AOPC) metric to mathematically prove that the highlighted evidence causally drove the model's detection decision.

## 🏗️ System Architecture
The system operates through four highly integrated analytical layers:
1. **Voice Detection Engine:** Ingests raw audio waveforms and outputs a fused classification score.
2. **Explainable Spoof Voice Analysis System:** Non-invasively extracts internal representations from the voice engine to generate visual and semantic evidence.
3. **Data-Centric AI Text Detection Engine:** Cleans raw text using an LLM-driven agent and processes it through a dual-objective fine-tuned DeBERTa-v3 transformer.
4. **Hybrid Neural Interpretability Framework (HNIF) & Agentic SDK:** Generates mathematically grounded, natural-language evidence reports for text classification decisions.

## 🧑‍💻 Module Breakdown & Contributors

### 1. Multi-Branch Deepfake Voice Detection System
**Contributor:** Aweesha Thavishanka Wijesundara (IT22183668)  
Processes input audio through four specialized detection branches to maximize robustness against unseen spoofing and voice-cloning techniques.
* **Spectral Analysis:** Uses LFCC combined with CNN/TCN architectures.
* **Spectro-Temporal Modeling:** Employs the AASIST graph-based architecture.
* **Long-Range Temporal Coherence:** Integrates SSL speech representations (WavLM/XLSR) with BiLSTM/Mamba sequence modeling.
* **Physiological Analysis:** Captures human voice production cues using glottal source features (IAIF, NAQ, QOQ).
* **Fusion:** Combines all independent branches using an adaptive score-level fusion mechanism.

### 2. Data-Centric AI Text Detection Engine
**Contributor:** Ishara Udayanga Athapaththu (IT22251046)  
A robust and equitable text classification pipeline designed to defeat adversarial paraphrasing attacks while protecting innocent ESL writers.
* **Agentic Pre-processing:** Uses a LlamaIndex-driven ReAct Agent to clean text and neutralize adversarial tactics (e.g., zero-width spaces, Cyrillic homoglyphs) prior to tokenization.
* **Disentangled Attention:** Leverages the DeBERTa-v3 architecture to separate semantic content from syntactic positioning.
* **Mixed-Domain Fine-Tuning:** Trained using weighted random sampling on a balanced dataset combining the RAID adversarial benchmark and authentic ESL academic writing to minimize the False Positive Rate (FPR).

### 3. Explainable Spoof Voice Analysis System (ESVAS)
**Contributor:** Silva S. P. S. (IT22219602)  
Provides transparent, auditable evidence for voice classification decisions without altering the upstream classifier's inference graphs.
* **Temporal Attention Visualizer:** Uses Attention Rollout on the XLSR encoder to mathematically highlight anomalous time-frequency regions directly on an audio spectrogram.
* **Semantic Feature Explainer:** Trains an XGBoost surrogate model on interpretable acoustic features (jitter, shimmer, glottal parameters) and applies exact SHAP analysis to quantify their predictive impact.
* **Evaluation & Synthesis:** Calculates spatial consistency (IoU) and explanation faithfulness (AOPC) metrics, delivering insights via an interactive investigative dashboard.

### 4. Hybrid Neural Interpretability Framework (HNIF) & Agentic SDK
**Contributor:** Dinitha Fernando (IT22090744)  
Translates the text detector's complex internal mathematics into natural-language, evidence-based audit reports.
* **Hybrid Saliency Fusion:** Mathematically multiplies multi-layer Attention Rollout with Integrated Gradients to filter out syntactic noise and isolate highly causal AI tokens.
* **Linguistic Smoothing:** Applies strict sub-word aggregation and Part-of-Speech (POS) masking to convert abstract token-level data into coherent linguistic markers.
* **Agentic Translation:** A specialized LLM agent interprets the JSON-structured saliency arrays, outputting a human-readable, transparent analytical summary.
* **Automated Validation:** Validates the generated explanations dynamically through real-time perturbation deletion testing (AOPC).

## 🛠️ Technologies Used
* **Deep Learning Frameworks:** PyTorch, Hugging Face Transformers
* **Architectures & Models:** DeBERTa-v3, XLSR-53, AASIST, CNN/TCN, BiLSTM/Mamba, XGBoost
* **Explainability Tools (XAI):** Captum (Integrated Gradients), SHAP (TreeExplainer), Attention Rollout
* **Agentic & NLP Tools:** LlamaIndex, spaCy, Gemini / OpenAI APIs
* **Audio Processing:** Librosa, Parselmouth (Praat API), IAIF Toolkit
* **Frontend & Visualization:** Streamlit, Matplotlib, React.js

## 🗄️ Datasets
* **Audio Repositories:** ASVspoof 2019/2021 LA, FakeSound2, PartialSpoof
* **Text Repositories:** RAID Benchmark (Adversarial attacks), Custom Curated ESL Corpus

## 🚀 Installation & Setup

```bash
# 1. Clone the repository
git clone [https://github.com/your-org/R26-038-Deepfake-Detection.git](https://github.com/your-org/R26-038-Deepfake-Detection.git)
cd R26-038-Deepfake-Detection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Add your LLM API keys for the Agentic Translation SDK and ReAct Preprocessor
echo "LLM_API_KEY=your_api_key_here" > .env

# 5. Run the backend inference server
uvicorn app.main:app --reload

# 6. Launch the Interactive Analytical Dashboard
streamlit run frontend/dashboard.py

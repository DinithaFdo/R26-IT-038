text + xai# Explainable AI (XAI) Service Documentation & Implementation Tracker
## 1. Overview
Provides deep-learning interpretability by extracting causal mathematical weights.
- Architecture: Follows strict Domain-Driven Design (DDD) decoupling validation, extraction, and filtering.
## 2. Core Schemas & Validation (schemas.py)
Defines strict Pydantic schemas for request validation and serialization.
- XAIRequest: Input contract wrapping incoming text string for explainability processing.
- TokenScore: Schema capturing token string, raw score, normalized score, and noise flag.
- XAIResponse: Output contract containing predicted_class, confidence, heatmap_data, and interpretability_report.
- Swagger UI: Auto-generated OpenAPI v3 schemas ensure seamless API contract verification.
## 3. Mathematical Extraction Engine (core_extractor.py)
Extracts transformer attention and gradient attributions.
- Attention Extraction: Captures multi-head attention weights from the final transformer encoder layer.
- Attention Aggregation: Averages attention heads across tokens to produce a unified 1D vector.
- Integrated Gradients: Utilizes Captum LayerIntegratedGradients targeting the word embedding layer.
- Baseline Strategy: Uses zero/pad token embedding tensor baselines for path-integral calculation.
- Riemann Approximation: Computes m-step path integrals (default n_steps=50) for numerical stability.
- Vector Norm: Applies Euclidean L2-norm across hidden dimensions to convert embedding gradients to scalar token scores.
- Hybrid Fusion Equation: Score = Final Layer Base Attention * Captum Integrated Gradients.
- Target Class Attribution: Selects the predicted class logit index as the target for gradient backpropagation.
- Attribution Clamping: Evaluates positive causal attribution to prevent negative cancellation.
- Min-Max Normalization: Rescales attribution scores into [0.0, 1.0] interval for heatmap rendering.
- Subword Reconstruction: Handles subword tokens by matching offsets to full lexical tokens.
## 4. NLP & Linguistic Filtering (linguistic_filter.py)
Applies linguistic and part-of-speech filtering to eliminate non-causal grammatical noise.
- spaCy Model: Uses en_core_web_sm pipeline with parser and tagger enabled.
- POS Filter (PUNCT): Marks punctuation tokens (commas, periods, quotation marks) as noise.
- POS Filter (CCONJ): Filters coordinating conjunctions (and, but, or) to prevent spurious importance.
- POS Filter (DET): Masks definite and indefinite articles (the, a, an) as non-causal tokens.
- POS Filter (ADP): Filters prepositions and adpositions (in, to, for, with).
- POS Filter (SPACE): Identifies and suppresses newline and space tokens from heatmap consideration.
- POS Filter (PART): Eliminates grammatical particles (e.g., possessive markers, negative particles).
- Score Suppression: Sets normalized_score = 0.0 and is_noise = True for all filtered tokens.
- Subword Merging: Strips subword artifacts (e.g. ## or byte-pair markers) before linguistic analysis.
- Semantic Saliency: Preserves content words (NOUN, VERB, ADJ, ADV) to isolate genuine causal triggers.
## 5. Agentic LLM Translation Layer (llm_translator.py)
Translates raw attribution tensors into human-readable forensic reports.
- LLM Provider: Integrates Google Gemini API via official SDK client.
- Configuration: Reads GEMINI_API_KEY from environment variables with graceful fallback validation.
- Top-K Token Selection: Extracts the top-5 highest-scoring non-noise tokens to construct the prompt.
- System Prompt: Instructs the model to act as a forensic AI interpretability auditor.
- Anti-Hallucination: Strictly constrains generated explanations to provided salient tokens.
- Report Length Constraint: Enforces a concise 3-sentence summary targeting non-technical stakeholders.
- Fault Tolerance: Implements template-based deterministic fallback when API limits or errors occur.
- Generation Hyperparameters: Configures low temperature (0.2) and top_p for deterministic auditing.
## 6. Service Orchestration Pipeline (service.py)
Coordinates end-to-end execution of classification, extraction, and explanation.
- State Extraction: Accesses app.state.model and app.state.tokenizer without redundant reloads.
- Step 1 (Inference): Runs forward pass to compute class logits and softmax probabilities.
- Step 2 (Classification): Extracts argmax label and percentage confidence score.
- Step 3 (Feature Extraction): Invokes core_extractor.extract_hybrid_attribution.
- Step 4 (Filtering): Invokes linguistic_filter.apply_pos_mask to sanitize raw tokens.
- Step 5 (Translation): Invokes llm_translator.generate_audit_report with top salient tokens.
- Step 6 (Response Packaging): Compiles results into verified XAIResponse payload.
- Error Handling: Catches inference errors, logs diagnostics, and raises formatted HTTPException.
## 7. FastAPI API Routing (router.py)
Exposes RESTful endpoints for explainability auditing.
- Endpoint: POST /xai/audit accepting JSON payload with request validation.
- Dependency Injection: Injects Request object to safely access application state singletons.
- Pre-Flight Verification: Validates that PyTorch model and tokenizer are initialized in memory.
- Response Annotation: Binds response_model=XAIResponse with 200, 400, 500 error mappings.
## 8. Memory Optimization & Resource Sharing
Zero-duplication architecture sharing PyTorch weights across modules.
- Lifespan Management: Loads transformer weights once in app.py lifespan context manager.
- VRAM Efficiency: Avoids loading duplicate weights for both classification and XAI.
- Inference Optimization: Sets model.eval() to disable dropout layers during attribution.
- Gradient Scoping: Dynamically enables gradient tracking only inside Captum attribution scope.
- Hardware Adaptability: Detects CUDA availability and seamlessly defaults to CPU execution.
## 9. Containerization & Docker Deployment
Containerized deployment separating code artifacts from heavy binary model weights.
- Volume Mount Strategy: Mounts external ./models directory directly into container at /app/models.
- Volume Security: Enforces :ro flag to prevent container runtime from modifying checkpoint files.
- Git Exclusions: Prevents binary files (*.safetensors, *.bin) from polluting git history.
- Dockerignore: Skips heavy model weight directories during Docker build context transfer.
- Dockerfile Architecture: Multi-stage python-slim base image minimizing overall image size.
- Compose Specification: Exposes port 8000 and connects to shared internal microservice network.
- Healthcheck: Configures curl probe on /health endpoint to monitor worker readiness.
- Environment Binding: Passes GEMINI_API_KEY, MODEL_PATH, and LOG_LEVEL via compose environment.
## 10. Security & Input Sanitization
Validates input bounds and protects against adversarial payload processing.
- Token Truncation: Restricts text input to maximum sequence length (512 tokens) to bound VRAM usage.
- Character Sanitization: Strips zero-width spaces and malicious unicode bypass sequences.
- Rate Limiting: Safeguards compute-intensive interpretability queries with sliding window limits.
- Error Masking: Wraps internal exceptions in safe client-facing error structures.
## 11. Testing & Verification Suite
Comprehensive unit and integration test strategies for interpretability.
- Test Fixtures: Provides lightweight dummy PyTorch transformer for offline unit testing.
- POS Unit Tests: Verifies that punctuation, prepositions, and stop-words receive zero attribution.
- Normalization Tests: Asserts all normalized scores strictly reside within [0.0, 1.0].
- LLM Mocks: Uses unittest.mock to simulate Gemini responses without network calls.
- Integration Tests: FastAPI TestClient verifies end-to-end audit request and response schema.
## 12. Performance Benchmarks & Latency Profiling
Profiling execution latency across extraction, filtering, and LLM translation.
- Inference Latency: Raw classification forward pass takes ~25ms on GPU.
- Extraction Latency: Captum 50-step path integral takes ~180ms.
- spaCy Filtering Latency: Token POS tagging and masking executes in ~8ms.
- LLM Translation Latency: Remote Gemini API call consumes ~450ms.
- Total Turnaround: Complete forensic audit completes within ~660ms.
- Async Concurrency: Runs LLM translation asynchronously to prevent event loop blocking.
- Thread Offloading: Offloads synchronous PyTorch computations to background thread pool.
## 13. Frontend UI Integration
Specifications for visualizing token saliency on web dashboards.
- Color Scale: Uses dynamic RGBA background interpolation: rgba(239, 68, 68, score).
- Interactive Tooltip: Displays raw score, normalized score, and POS tag on token hover.
- Audit Card: Renders 3-sentence summary in a dedicated callout card with confidence badge.
- Accessibility: Implements accessible high-contrast text styling over saturated token backgrounds.
## 14. Future Improvements & Roadmap
Extending interpretability framework to multi-modal audio and image tasks.
- SHAP Comparison: Evaluating KernelSHAP benchmarks against Integrated Gradients.
- Cross-Attention: Extending HNIF framework to encoder-decoder sequence generation models.
- Explanation Caching: Redis key-value cache keyed by SHA-256 text hash for instant lookups.
- Fairness Audits: Analyzing token importance distribution across demographic datasets.
- CI/CD Integration: GitHub Actions workflow running pytest and black/flake8 on every PR.

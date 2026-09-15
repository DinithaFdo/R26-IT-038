# Deployment Guide for the AI Detection Backend

This guide is for a production deployment on Linux-based infrastructure such as RunPod, a GPU VM, or a container host.

## 1. Deployment goals

The backend should be:

- fast and low latency
- secure
- stateless where possible
- able to reuse large model files without storing them in Git
- able to auto-rebuild when the repo changes

## 2. Recommended architecture

### Best option for low latency

Use a persistent model volume mounted into the container at runtime.

Recommended layout:

```text
/workspace/
├── app/
│   └── ai-detection-app/
│       ├── app.py
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── .env
│       └── ...
└── models/
    ├── branch1_deberta_large_model2_domainfix_final/
    ├── branch2_lexical_model_v4_domainfix/
    ├── deberta_temperature.json
    └── ...
```

Then mount:

```text
/workspace/models:/app/models:ro
```

This gives the container read-only access to model weights while keeping them outside the Git repo.

### Why this is best

- no giant Docker image bloat
- faster redeploys
- no secret model files in GitHub
- lower storage cost on app image builds
- easier to change models without rebuilding the app

## 3. Model storage strategy

### Preferred for this project

Store model files on a persistent volume or a dedicated object-store mount.

Use these rules:

- never commit model files to Git
- never bake model weights into the Docker image
- keep model folder outside the repo
- map model folder to `/app/models` inside the app container

### Example Linux environment values

```env
MODEL2_PATH=/app/models/branch1_deberta_large_model2_domainfix_final
XGB_MODEL_PATH=/app/models/branch2_lexical_model_v4_domainfix/model.pkl
XGB_SCALER_PATH=/app/models/branch2_lexical_model_v4_domainfix/scaler.pkl
XGB_FEATURES_PATH=/app/models/branch2_lexical_model_v4_domainfix/feature_names.pkl
XGB_CALIBRATOR_PATH=/app/models/branch2_lexical_model_v4_domainfix/isotonic_calibrator.pkl
DEBERTA_TEMPERATURE_PATH=/app/models/deberta_temperature.json
CSS_MIN_WORD_COUNT=150
GEMINI_API_KEY=your-real-key
ALLOWED_ORIGINS=https://your-domain.com
HOST=0.0.0.0
PORT=8000
```

### Do not use this pattern

Avoid: local Windows paths like `G:\My Drive\...` in production or in a Linux container.

## 4. Dependencies and package hygiene

The project was originally written for a Google Colab style environment, and some setup assumptions are not safe for a production server.

### This project should not use Colab packaging assumptions in production

The original requirements file had comments like "torch is provided by Colab" and similar notes. That is not appropriate for a server environment.

Instead:

- install exact runtime packages for the target OS
- install the correct Torch build for the target GPU or CPU
- keep dependency versions constrained to avoid conflicts
- avoid `bitsandbytes` and `accelerate` unless you specifically need them

### Production-safe dependency rule

Use a Linux GPU image with the corresponding CUDA build of PyTorch, then install Python packages in the image or in the runtime environment.

## 5. Container design

The app is already structured to work as a container service:

- [app.py](app.py) creates the FastAPI app
- [Dockerfile](Dockerfile) builds the Python service
- [docker-compose.yml](docker-compose.yml) mounts the model directory
- [.dockerignore](.dockerignore) keeps large files out of build context
- [.gitignore](.gitignore) keeps model data out of Git

This is the correct pattern for a deployment pipeline.

## 6. How to run locally for deployment testing

On a Linux machine or Docker-enabled machine:

```bash
docker compose up --build
```

Then verify:

```bash
curl http://localhost:8000/
curl http://localhost:8000/classify/health
```

Open:

- http://localhost:8000/docs

## 7. How to deploy on RunPod

Recommended approach:

1. Create a GPU pod or a persistent environment on RunPod.
2. Clone the repo into the pod.
3. Mount a persistent volume for `/app/models`.
4. Copy model files into the mounted volume.
5. Set environment variables in the pod or via secret manager.
6. Build the container image or run the app in a Docker container.
7. Expose the backend port, usually 8000.

### Persistent volume pattern

```bash
mkdir -p /workspace/models
# copy model folders here
```

Then mount to the container:

```bash
/workspace/models:/app/models:ro
```

This is the fastest and most stable pattern for your project.

## 8. Secure production setup

Use the following production standards:

- keep secrets in environment variables or a secret manager
- do not hard-code API keys in source files
- use CORS allowlist only for trusted domains
- do not expose the app publicly unless using a proxy or firewall
- run the app on a non-root user inside the container
- add health checks and restart policies
- keep logs centralized

## 9. CI/CD flow

### GitHub workflow model

You can use GitHub Actions to automate deployment:

1. push code to GitHub main branch
2. workflow checks out the repo
3. workflow builds the Docker image
4. workflow pushes image to container registry
5. workflow updates the deployment environment with the new image or restarts the pod

### What changes when you update the repo

When a code change is pushed:

- the CI pipeline rebuilds the app image
- the new image is pushed to your registry
- deployment automation restarts the backend or replaces the container
- model files are not re-uploaded because they stay on the mounted model volume

This is the right approach for a model-heavy backend.

## 10. Recommended deployment pattern for this project

### Option A: Best for speed and simplicity

- GitHub repo holds app code only
- model files stored on a persistent volume mounted into the server
- Docker image runs the backend
- RunPod host or VM mounts the same model directory at runtime

### Option B: Better for large teams

- GitHub repo holds code
- models stored in a dedicated object store or managed volume
- CI/CD pipeline builds and deploys the app image
- deployment uses environment variables for model path and keys

## 11. Example GitHub Actions workflow

See [.github/workflows/deploy.yml](.github/workflows/deploy.yml) for a sample deployment workflow.

## 12. Important operational notes

- Do not run `pip install` directly inside the repo on your laptop for production validation. Use Docker or a clean target environment instead.
- Keep this repo code-only, as the model folder is already excluded.
- If you change the app code, the deployment should rebuild the image; if you change model files, only the mounted volume needs to be updated.
- Use `/app/models` as the canonical model mount path inside the container.

## 13. Final recommendation

For this project, the enterprise-grade approach is:

- GitHub for code
- Docker for runtime packaging
- persistent Linux volume for models
- env secrets for keys and endpoints
- CI/CD pipeline to rebuild and redeploy on code changes
- RunPod or a GPU VM as the cloud runtime

This is the safest and lowest-latency deployment pattern for your current architecture.

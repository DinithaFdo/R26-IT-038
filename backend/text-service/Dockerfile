# Use the official Python slim image for a balance of size and compatibility
FROM python:3.11-slim

# Set environment variables for Python optimization
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# PYTHONUNBUFFERED: Ensures logs are outputted immediately without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MPLCONFIGDIR=/tmp/matplotlib

# Create a non-root user and group for security (Production Best Practice)
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser

# Set the working directory inside the container
WORKDIR /app

# Ensure the non-root user can write to Matplotlib temp cache paths
RUN mkdir -p /tmp/matplotlib && chown -R appuser:appgroup /tmp/matplotlib

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies and clean up cache to keep the image small
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# CRITICAL: Download the spaCy NLP model required by xai/linguistic_filter.py
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code into the container
COPY . .

# Change ownership of the working directory to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user before running the application
USER appuser

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
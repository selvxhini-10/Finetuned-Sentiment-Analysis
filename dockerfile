# 1. Lightweight Python base image (matches training Python version)
FROM python:3.11.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (minimal, required for HF + Torch)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy dependency file first (enables Docker layer caching)
COPY requirements.txt .

# 5. Install Python dependencies (CPU-only)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6. Copy application code (inference + API only)
COPY app /app/app
COPY inference /app/inference

# 7. Copy serving model artifacts
COPY trained_model /app/trained_model

# 8. Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 9. Expose FastAPI port
EXPOSE 8080

# 10. Start FastAPI using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
# --- Stage 1: Build Stage ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal compilers for pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. We install to /install to keep the final image clean
# 2. We use --no-cache-dir to save space during the build process
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime Stage ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install ONLY the C-libraries OpenCV needs to run (without the GUI bloat)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the pre-installed dependencies from the builder
COPY --from=builder /install /usr/local

# Copy your application code (ensure api/ and utils/ are NOT in .dockerignore)
COPY . .

# Set working directory to where main.py is
WORKDIR /app/api

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
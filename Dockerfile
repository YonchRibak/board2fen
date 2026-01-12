# Dockerfile
FROM python:3.11-slim

# Faster, cleaner Python logs and no .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for psycopg2, Pillow/OpenCV runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    build-essential gcc \
  && rm -rf /var/lib/apt/lists/*

# Base workdir
WORKDIR /app

# Install lightweight runtime deps only (use your split requirements.txt)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the whole repo
COPY . /app

# Your code lives in api/, and main imports 'config' from the same folder.
# Run from /app/api so "from config import settings" resolves.
WORKDIR /app/api

EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

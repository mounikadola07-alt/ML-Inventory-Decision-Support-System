# Dockerfile for Enterprise Inventory Decision Support System
# Provides a completely isolated, reproducible computational environment.

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest pytest-mock python-docx python-pptx

# Copy application code, tests, and data
COPY . /app

# Run full automated test suite by default
CMD ["python", "-m", "pytest", "tests/", "-v"]

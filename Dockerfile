# ==============================================================================
# FlowCast: AI-Powered Smart Traffic Flow Prediction Platform
# Container Image for FastAPI Backend & Streamlit Interactive Dashboard
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, dashboards, serialized models, and documentation
COPY src/ /app/src/
COPY dashboard/ /app/dashboard/
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/
COPY docs/ /app/docs/
COPY tests/ /app/tests/

# Expose API and Streamlit ports
EXPOSE 8000 8501

# Health check default against FastAPI
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command launches FastAPI backend
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

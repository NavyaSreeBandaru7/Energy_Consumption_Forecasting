# Energy Forecasting Platform - Main Application Dockerfile
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies for production
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/cache && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . /app/

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Wait for database\n\
if [ "$DB_HOST" ]; then\n\
  echo "Waiting for database..."\n\
  while ! nc -z $DB_HOST ${DB_PORT:-5432}; do\n\
    sleep 1\n\
  done\n\
  echo "Database started"\n\
fi\n\
\n\
# Wait for Redis\n\
if [ "$REDIS_HOST" ]; then\n\
  echo "Waiting for Redis..."\n\
  while ! nc -z $REDIS_HOST ${REDIS_PORT:-6379}; do\n\
    sleep 1\n\
  done\n\
  echo "Redis started"\n\
fi\n\
\n\
# Initialize application\n\
echo "Initializing Energy Forecasting Platform..."\n\
python -c "import energy_forecasting_main; print('\''Application modules loaded successfully'\')"\n\
\n\
# Start the application\n\
echo "Starting Energy Forecasting Platform..."\n\
exec "$@"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Install netcat for health checks
RUN apt-get update && apt-get install -y netcat-traditional && rm -rf /var/lib/apt/lists/*

# Switch to app user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose ports
EXPOSE 8501

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["streamlit", "run", "energy_forecasting_main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

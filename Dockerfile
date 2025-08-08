# Multi-stage build for production Meulex deployment
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -r meulex && useradd -r -g meulex meulex

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY --chown=meulex:meulex . .

# Install the application in development mode
RUN pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && chown -R meulex:meulex /app/data /app/logs

# Switch to non-root user
USER meulex

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["meulex", "api", "--host", "0.0.0.0", "--port", "8000"]

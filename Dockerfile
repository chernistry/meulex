# Multi-stage build for production Meulex deployment
FROM python:3.12-slim as builder

# Set build arguments
ARG POETRY_VERSION=1.8.3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Set Poetry configuration
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN groupadd -r meulex && useradd -r -g meulex meulex

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=meulex:meulex /app/.venv /app/.venv

# Copy application code
COPY --chown=meulex:meulex . .

# Install the application
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
CMD ["meulex", "api", "start", "--host", "0.0.0.0", "--port", "8000"]

# Development stage (optional)
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
COPY --from=builder /app/.venv /app/.venv
RUN pip install pytest pytest-cov pytest-asyncio black isort mypy

# Switch back to meulex user
USER meulex

# Override command for development
CMD ["meulex", "api", "start", "--host", "0.0.0.0", "--port", "8000", "--reload"]

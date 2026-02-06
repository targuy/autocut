# Multi-stage Dockerfile for AutoCut-Agent

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.0

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Export requirements (poetry can be slow in containers)
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Stage 2: Runtime
FROM python:3.10-slim

LABEL maintainer="AutoCut Team"
LABEL description="AutoCut-Agent - Intelligent task orchestration system"

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 agent && \
    chown -R agent:agent /app

# Copy requirements from builder
COPY --from=builder /build/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=agent:agent src/ ./src/
COPY --chown=agent:agent configs/ ./configs/
COPY --chown=agent:agent scripts/docker-entrypoint.sh /entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/outputs && \
    chown -R agent:agent /app/data /app/logs /app/outputs

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER agent

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    AGENT_LOG_LEVEL=INFO

# Volume for persistent data
VOLUME ["/app/data", "/app/logs", "/app/outputs"]

# Entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["start", "--config", "/app/configs/default.yaml"]

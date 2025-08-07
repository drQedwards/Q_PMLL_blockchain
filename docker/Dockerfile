# PMLL Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .[all]

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY scripts/ ./scripts/

# Create directories for data and configs
RUN mkdir -p /app/data /app/config /app/logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PMLL_HOST=0.0.0.0
ENV PMLL_PORT=8001
ENV PMLL_DEVICE=cpu

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Default command
CMD ["python", "scripts/run_pmll_server.py", "--host", "0.0.0.0", "--port", "8001"]
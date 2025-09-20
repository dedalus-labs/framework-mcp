# Dockerfile for Dedalus Documentation MCP Server
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY docs/ ./docs/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DOCS_DIR=/app/docs

# Run the server
CMD ["python", "src/server.py"]
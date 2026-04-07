# AMPIDE — Dockerfile
# Runs on CPU: 2 vCPU, RAM: 8GB
# Compatible with Hugging Face Spaces

FROM python:3.11-slim

# Non-root user for security
RUN useradd -m -u 1000 ampide

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY ampide_env.py .
COPY tasks.py      .
COPY grader.py     .
COPY server.py     .
COPY inference.py  .
COPY openenv.yaml  .

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Switch to non-root
USER ampide

ENV AMPIDE_DEFAULT_TASK=easy_direct_injection
ENV PYTHONUNBUFFERED=1

CMD ["python", "server.py"]

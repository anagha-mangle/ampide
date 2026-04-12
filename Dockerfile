FROM python:3.11-slim

RUN useradd -m -u 1000 ampide

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY ampide_env.py .
COPY tasks.py      .
COPY grader.py     .
COPY inference.py  .
COPY openenv.yaml  .
COPY server ./server   # ✅ FIXED

EXPOSE 7860

USER ampide

ENV AMPIDE_DEFAULT_TASK=easy_direct_injection
ENV PYTHONUNBUFFERED=1

RUN pip install uv
RUN python -m uv sync

CMD ["python", "-m", "server.app"]

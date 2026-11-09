FROM python:3.13-slim

WORKDIR /app

# Only the "api" dependency group: training libraries (torch, jupyter) stay out of the image.
COPY pyproject.toml .
RUN pip install --no-cache-dir --group api

COPY app/ ./app/

# Prometheus multiprocess mode (PROMETHEUS_MULTIPROC_DIR, set in docker-compose.yaml): the
# metrics folder must be emptied before the workers start, or old counts carry over.
CMD ["sh", "-c", "if [ -n \"$PROMETHEUS_MULTIPROC_DIR\" ]; then rm -rf \"$PROMETHEUS_MULTIPROC_DIR\" && mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\"; fi && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 8"]

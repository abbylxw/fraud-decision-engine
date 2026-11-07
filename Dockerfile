FROM python:3.13-slim

WORKDIR /app

# Only the "api" dependency group: training libraries (torch, jupyter) stay out of the image.
COPY pyproject.toml .
RUN pip install --no-cache-dir --group api

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "8"]

FROM python:3.11-slim

# System deps: tesseract for OCR (DR-06), build tools for native wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Ingest the sample corpus at build time so the image is queryable out of the box.
# In production, mount data/pdfs and run `python -m scripts.ingest` instead.
RUN python -m scripts.load_sample_corpus || true

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

# System deps for PyMuPDF / Tesseract
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Create data & vectorstore dirs
RUN mkdir -p data vectorstore eval

# Expose ports: 8000 (API) + 8501 (Streamlit)
EXPOSE 8000 8501

# Default: start both API and Streamlit
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]

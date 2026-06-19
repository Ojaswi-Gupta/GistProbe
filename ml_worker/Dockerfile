FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK and SpaCy models during the build process
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" && \
    python -m spacy download en_core_web_sm

COPY . .

# Expose port for HuggingFace Spaces (usually 7860)
EXPOSE 7860

# Command to run the application using Uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port 7860

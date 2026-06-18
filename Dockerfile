# Use an official Python runtime as a parent image
# We use the 'slim' version to reduce container size, but it includes what we need.
FROM python:3.11-slim

# Set environment variables to ensure Python runs optimally in a container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for Playwright, SpaCy, and WordCloud (fonts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    fonts-liberation \
    fontconfig \
    # Dependencies often needed for headless Chrome/Playwright
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache for dependencies
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK and SpaCy models during the build process
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" && \
    python -m spacy download en_core_web_sm

# Install Playwright's headless Chromium browser
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application using Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "--timeout", "120", "app:app"]

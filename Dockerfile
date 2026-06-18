# Use the official Microsoft Playwright image which comes pre-installed with Chromium and ALL its Linux dependencies.
# This completely bypasses the need for "playwright install-deps" which causes errors on newer Debian systems.
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# We only need to install fonts-liberation for WordCloud (Playwright dependencies are already handled)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-liberation \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK and SpaCy models during the build process
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" && \
    python -m spacy download en_core_web_sm

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application using Gunicorn for production
# We use the shell form so that it evaluates the $PORT environment variable injected by Render.
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 3 --timeout 120 app:app

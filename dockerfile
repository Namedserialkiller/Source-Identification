FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for xgboost)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

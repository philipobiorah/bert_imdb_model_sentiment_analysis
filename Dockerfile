# Use Python 3.10 for compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create writable directories
RUN mkdir -p /tmp/huggingface_cache /tmp/matplotlib \
    && chmod -R 777 /tmp/huggingface_cache /tmp/matplotlib

# Set cache environment variables
ENV HF_HOME=/tmp
ENV TRANSFORMERS_CACHE=/tmp
ENV MPLCONFIGDIR=/tmp

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Install PyTorch separately for compatibility
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy application files
COPY . /app

# Expose Flask app port
EXPOSE 7860

# Run the application
CMD ["python", "main.py"]

FROM python:3.9-slim-buster

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the files into the container
COPY . .

# Install Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

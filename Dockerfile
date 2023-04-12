FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y python3-pip
# Copy the files into the container
COPY . .
RUN pip3 install --upgrade pip

# Install Python packages specified in requirements.txt
RUN pip3 install -r requirements.txt

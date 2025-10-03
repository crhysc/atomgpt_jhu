# Use Ubuntu base image and install Miniforge manually
FROM ubuntu:22.04

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV CONDA_ENV_NAME=my_atomgpt
ENV PYTHONUNBUFFERED=1

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniforge
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O miniforge.sh && \
    bash miniforge.sh -b -p /opt/conda && \
    rm miniforge.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Create conda environment with Python 3.10
RUN conda create --name ${CONDA_ENV_NAME} python=3.10 -y

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "my_atomgpt", "/bin/bash", "-c"]

# Clone AtomGPT repository
RUN git clone https://github.com/atomgptlab/atomgpt.git && \
    cd atomgpt && \
    pip install -e .

# Set the default working directory to atomgpt
WORKDIR /workspace/atomgpt

# Activate conda environment by default
RUN echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc

# Default command
CMD ["/bin/bash"]


###################################
# Instructions
#docker build -t atomgpt .
#docker login
#docker tag atomgpt atomgptlab/atomgpt:latest
#docker push atomgptlab/atomgpt:latest
#docker pull atomgptlab/atomgpt:latest

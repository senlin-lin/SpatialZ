# Use an official Python runtime as a parent image with CUDA support from NVIDIA
FROM nvidia/cuda:11.7.1-base-ubuntu18.04
# OR
#FROM nvcr.io/nvidia/cuda:11.7.1-base-ubuntu18.04

# Set the working directory in the container to /app
WORKDIR /app

# Set non-interactive installation mode
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages for building Python
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    liblzma-dev \
    llvm \
    libpq-dev

# Download and install Python 3.9.19
RUN wget https://www.python.org/ftp/python/3.9.19/Python-3.9.19.tgz \
    && tar xzf Python-3.9.19.tgz \
    && cd Python-3.9.19 \
    && ./configure --enable-optimizations \
    && make altinstall

# Set python3.9 as the default python and pip
RUN ln -s /usr/local/bin/python3.9 /usr/bin/python3 \
    && ln -s /usr/local/bin/pip3.9 /usr/bin/pip3 \
    && ln -s /usr/local/bin/python3.9 /usr/bin/python \
    && ln -s /usr/local/bin/pip3.9 /usr/bin/pip

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Jupyter Notebook
RUN pip3 install notebook

# Copy the repository directory files to the container at /app
COPY . /app

# Copy the repository directory files to the container at /app
RUN pip3 install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html \
    numpy==1.22.2 \
    pandas==2.0.0 \
    anndata==0.9.1 \
    scikit-learn==1.2.2 \
    scipy==1.10.1 \
    pot==0.9.3 \
    SOMENDER==1.1 \
    open3d==0.18.0 \
    scanpy==1.9.3 \
    squidpy==1.2.3 \
    palettable \
    scikit-misc

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--notebook-dir=/app"]

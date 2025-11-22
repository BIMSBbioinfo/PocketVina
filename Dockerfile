FROM nvidia/cuda:12.3.0-devel-ubuntu22.04
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV TZ=UTC

# Configure apt sources
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo "deb http://de.archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://de.archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://de.archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# Install Python, build tools, and OpenCL development dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        git \
        wget \
        build-essential \
        cmake \
        openjdk-17-jdk \
        libopenblas-dev \
        ocl-icd-opencl-dev \
        ocl-icd-libopencl1 \
        opencl-headers \
        clinfo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Create Python symlinks
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python3 --version

# (Optional diagnostic: list available OpenCL libraries from the CUDA toolkit)
RUN find /usr/local/cuda -type f -name "libOpenCL.so*"

# Create an OpenCL ICD file that points to the host’s NVIDIA OpenCL ICD.
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "/usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

WORKDIR /app

# Clone and rearrange PocketVina
RUN git clone https://github.com/BIMSBbioinfo/PocketVina && \
    mv PocketVina/* . && \
    rm -rf PocketVina/

# Copy application source code (ensure you have a .dockerignore to exclude large files)
COPY . .

# Upgrade pip and build tools, build the wheel, and install it.
RUN python3 -m pip install --upgrade pip setuptools wheel build && \
    python3 -m build --wheel && \
    python3 -m pip install dist/*.whl && \
    python3 -m pip install --no-cache-dir --ignore-installed blinker && \
    python3 -m pip install --no-cache-dir --ignore-installed flask fastapi uvicorn python-multipart boto3 loguru pandas

# Set environment variables
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/cuda/bin:${PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    OPENCL_ROOT="/usr/local/cuda" \
    OPENCL_VENDOR_PATH="/etc/OpenCL/vendors" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    PATH="/app/build/lib/pocketvina/p2rank_2.5:${PATH}"

CMD ["python3"]

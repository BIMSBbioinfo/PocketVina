# PocketVina on NVIDIA GPUs (OpenCL through the NVIDIA container toolkit), with the web UI:
#   docker build --network=host -t pocketvina .
#   docker run --gpus all -it -p 7860:7860 pocketvina bash     # then: pocketvina --help, or python3 app/app.py
# The image holds the pocketvina wheel of this checkout (pip only copies files: the engine is the prebuilt
# pocketvina/PocketVina-GPU, rebuild it with `make -C pocketvina` before `docker build` if needed) and P2Rank 2.5.1
# in /opt/pocketvina, downloaded during the build by `pocketvina setup p2rank` (sha256-checked).
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04 AS base
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV TZ=UTC

# Configure apt sources
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo "deb http://de.archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://de.archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://de.archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# Python 3.10 (Ubuntu 22.04's own), Java 17 for P2Rank, the OpenCL ICD loader and headers, and build tools for
# rebuilding the engine in a container (the image holds no engine sources: `make -C pocketvina` in a checkout
# mounted into it, with that checkout's Boost)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-venv \
        python3-dev \
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
    python3 --version && \
    python3 -m pip install --no-cache-dir --upgrade pip

# (Optional diagnostic: list available OpenCL libraries from the CUDA toolkit)
RUN find /usr/local/cuda -type f -name "libOpenCL.so*"

# Create an OpenCL ICD file that points to the host’s NVIDIA OpenCL ICD.
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "/usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# The wheel, built in a stage of its own from only the files it is made of: BuildKit (the default builder) then
# sends nothing else of the checkout, such as benchmark data or the Boost/P2Rank trees of an older source install
FROM base AS wheel
COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE /src/
COPY pocketvina/*.py pocketvina/PocketVina-GPU /src/pocketvina/
COPY pocketvina/io/ /src/pocketvina/io/
COPY pocketvina/OpenCL/ /src/pocketvina/OpenCL/
RUN python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels /src

FROM base
COPY --from=wheel /wheels /wheels
# blinker first: flask (extra "app") cannot replace a blinker that apt installed
RUN python3 -m pip install --no-cache-dir --ignore-installed blinker && \
    python3 -m pip install --no-cache-dir "$(ls /wheels/pocketvina-*.whl)[app,bust,viz]" && \
    python3 -m pip install --no-cache-dir fastapi uvicorn python-multipart boto3 loguru && \
    rm -rf /wheels && \
    pocketvina --version

# P2Rank where every user of the container finds it (P2Rank writes nothing into its folder)
RUN pocketvina setup p2rank --dir /opt/pocketvina
ENV P2RANK_HOME=/opt/pocketvina/p2rank-2.5.1

WORKDIR /app
# the web UI (python3 app/app.py), the FastAPI service (cd fastapi && python3 main.py) and the example inputs.
# fastapi/ is not in git (.gitignore), so a clone has none: the pattern copies it only if the checkout has one
# (BuildKit, the default builder, allows a pattern that matches nothing)
COPY app/ app/
COPY fastap[i] fastapi/
COPY example/ example/
COPY README.md LICENSE ./

# Set environment variables
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/cuda/bin:${PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    OPENCL_ROOT="/usr/local/cuda" \
    OPENCL_VENDOR_PATH="/etc/OpenCL/vendors" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

CMD ["python3"]

ARG processor
ARG region
FROM 520713654638.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-tensorflow-scriptmode:1.12.0-cpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        jq \
        libav-tools \
        libjpeg-dev \
        libxrender1 \
        python3.6-dev \
        python3-opengl \
        wget \
        xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    PyOpenGL==3.1.0 \
    pyglet==1.3.2 \
    gym==0.12.5 \
    rl-coach-slim==1.0.0 && \
    pip install --no-cache-dir --upgrade sagemaker-containers && \
    pip install --upgrade numpy \
    pip install -U cmake \
    pip install --upgrade pip \
    pip install gym \
    pip install nes-py

WORKDIR /opt/ml

# Starts framework
ENTRYPOINT ["bash", "-m", "start.sh"]

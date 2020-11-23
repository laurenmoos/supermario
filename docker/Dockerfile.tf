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
    gym==0.17.3 \
    rl-coach-slim==1.0.0 && \
    pip install --no-cache-dir --upgrade sagemaker-containers && \
    pip install --upgrade numpy \
    pip install -U cmake \
    pip install --upgrade pip \
    pip install nes-py

WORKDIR /opt/ml

# Starts framework
ENTRYPOINT ["bash", "-m", "start.sh"]

# Patch Intel coach
COPY ./src/rl_coach.patch /opt/amazon/rl_coach.patch
RUN patch -p1 -N --directory=/usr/local/lib/python3.6/dist-packages/ < /opt/amazon/rl_coach.patch

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
    redis==2.10.6 \
    rl-coach-slim==1.0.0 && \
    pip install --no-cache-dir --upgrade sagemaker-containers && \
    pip install --upgrade numpy \
    pip install -U cmake \
    pip install --upgrade pip \
    pip install gym \
    pip install nes-py

ENV COACH_BACKEND=tensorflow

FROM coach-base:master as builder

# add coach source starting with files that could trigger
# re-build if dependencies change.
RUN mkdir /root/src
COPY setup.py /root/src/.
COPY requirements.txt /root/src/.
RUN pip3 install -r /root/src/requirements.txt

FROM coach-base:master
WORKDIR /root/src
COPY --from=builder /root/.cache /root/.cache
COPY setup.py /root/src/.
COPY requirements.txt /root/src/.
COPY README.md /root/src/.
]COPY . /root/src

# Copy workaround script for incorrect hostname
COPY lib/changehostname.c /
COPY lib/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

WORKDIR /opt/ml

# Starts framework
ENTRYPOINT ["bash", "-m", "start.sh"]

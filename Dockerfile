
ARG UBUNTU_VERSION=20.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=11.8.0
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} 

ARG CUDA_DOCKER_ARCH=all

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install \
    python3.9-dev \
    python3.9-venv \
    python3.9-distutils \
    tmux \
    curl \
    wget \
    vim \
    nano \
    ffmpeg \
    zip \
    unzip \
    git \
    espeak-ng \
    -y



# Set Python 3.9 as the default python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3

# install pip

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

RUN python3 get-pip.py

RUN python3 -m pip install --upgrade pip


WORKDIR /app

RUN git clone https://github.com/yl4579/StyleTTS2.git /app/tts

WORKDIR /app/tts

RUN python3.9 -m venv .venv



RUN mkdir -p /app/tts/Models/LibriTTS

RUN wget https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml -O /app/tts/Models/LibriTTS/config.yml
RUN wget https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth -O /app/tts/Models/LibriTTS/epochs_2nd_00020.pth

RUN . .venv/bin/activate && \
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U
COPY ./requirements.txt /app/tts/requirements.txt
COPY ./requirements-docker.txt /app/tts/requirements-docker.txt
RUN . .venv/bin/activate && \
pip install -r requirements.txt && \
pip install -r requirements-docker.txt

COPY ./launch.sh /app/launch.sh
RUN chmod +x /app/launch.sh


COPY ./tts-server.py /app/tts/tts-server.py

COPY ./Samples/ /app/tts/Samples/

EXPOSE 44332
EXPOSE 8767

ENTRYPOINT [ "/app/launch.sh" ]
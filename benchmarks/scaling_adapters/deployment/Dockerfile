FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install python3.9 python3.9-dev python3.9-distutils python3-pip -y && \
    ln -sf /usr/bin/python3.9 /usr/bin/python

WORKDIR /workspace
COPY / .
RUN python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN python -m pip install -e .

ENTRYPOINT ["python", "benchmarks/launch_server.py"]
EXPOSE 8000/tcp

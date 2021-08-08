FROM ubuntu:20.04

# from parent directory multilingual_kws:
# docker build -t mkws docker/ 
# docker run --rm -v $(pwd):/context -it mkws /bin/bash

# if you have nvidia-container-toolkit installed:
# docker run --rm --gpus all -v $(pwd):/context -it mkws /bin/bash

# switch to bash within the container
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND noninteractive
RUN apt update \
    && apt install --yes --no-install-recommends \
        fontconfig \
        unzip \
        locales\
        wget \
        curl \ 
        sox \ 
        ffmpeg \
        software-properties-common \
        git \
    && rm -rf /var/lib/apt/lists/*
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b &&  \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /miniconda/bin:$PATH

RUN conda update -y conda
RUN conda install -y python=3.7 

RUN git clone https://github.com/harvard-edge/multilingual_kws /multilingual_kws
WORKDIR /multilingual_kws
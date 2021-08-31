FROM tensorflow/tensorflow:2.3.4-gpu-jupyter

# from parent directory multilingual_kws:
# docker build -t mkws docker/ 

# inference can be run via:
# docker run --rm  -u $(id -u):$(id -g) -v $(pwd):/context -it mkws /bin/bash
# > python -m  embedding.run_inference --keyword hello --modelpath /context/hello_model --wav /context/hello_stream.wav
# if you have nvidia-docker installed, include the following flag to docker run: 
# --gpus all

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
        sox libsox-fmt-all libsox-dev \ 
        ffmpeg \
        software-properties-common \
        git \
        build-essential cmake \
        libmad0-dev \ 
        libid3tag0-dev libsndfile1-dev libgd-dev libboost-filesystem-dev \
        libboost-program-options-dev \
        libboost-regex-dev \
        vim \
    && rm -rf /var/lib/apt/lists/*
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# for bbc/peaks.js visualization
RUN git clone https://github.com/bbc/audiowaveform.git \
    && cd audiowaveform \
    && mkdir build \
    && cd build \
    && cmake -D ENABLE_TESTS=0 .. \
    && make \ 
    && make install
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - \
    && apt install -y nodejs \
    && npm install -g serve

RUN pip install dataclasses sox matplotlib seaborn pandas pydub fire 
RUN pip install tensorflow-io==0.16.0 --no-dependencies

# RUN echo "alpha-0.0.0"
RUN git clone https://github.com/harvard-edge/multilingual_kws /multilingual_kws
WORKDIR /multilingual_kws

FROM tensorflow/tensorflow:2.7.0-gpu
# for jupyter you can use the following:
# tensorflow/tensorflow:latest-gpu-jupyter

# from parent directory multilingual_kws:
# docker build -t mkws docker/ 

# inference and visualization can be run via the below command:
# note: if you do not have nvidia-docker installed, remove "--gpus all"

# docker run --gpus all -p 8080:8080 --rm -u $(id -u):$(id -g) -it \
#    -v $(pwd):/demo_data \
#    mkws \
#    --keyword mask \
#    --modelpath /demo_data/xfer_epochs_4_bs_64_nbs_2_val_acc_1.00_target_mask \
#    --groundtruth /demo_data/mask_groundtruth_labels.txt \
#    --wav /demo_data/mask_stream.wav \
#    --transcript /demo_data/mask_full_transcript.json \
#    --visualizer


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
        opus-tools \
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

# RUN echo "alpha-0.0.0"
RUN git clone https://github.com/harvard-edge/multilingual_kws /multilingual_kws
WORKDIR /multilingual_kws
RUN wget https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/multilingual_context_73_0.8011.tar.gz && \
    tar xf multilingual_context_73_0.8011.tar.gz
RUN wget https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/unknown_files.tar.gz && \
    tar xf unknown_files.tar.gz
RUN chmod -R 777 /multilingual_kws/visualizer/
ENTRYPOINT ["python", "-m", "multilingual_kws.run"]
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

## for apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive 
#set defaut shell to bash
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash 
ENV SHELL=/bin/bash

# Install basis C library, can be needed to build other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    wget \
    autoconf \
    tar \
    unzip \
    gcc-multilib \
    automake \
    sudo &&\
    rm -rf /var/lib/apt/lists/*

# install gdal
RUN add-apt-repository ppa:ubuntugis/ppa && \
    apt-get update

# Install python env
RUN apt-get -y install python3 python-is-python3 python3-pip && \
    python3 -m pip install --upgrade pip

# Install python requirements
COPY ./requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Setup user, avoid root user
RUN apt-get update && apt-get -y install gosu
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

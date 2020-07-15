FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
# FROM ubuntu:16.04
MAINTAINER Christian Schroeder de Witt

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git \
    libsm6 libxext6 libxrender-dev
#RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
#    rm -rf /var/lib/apt/lists/*


#Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install opencv-python
RUN pip3 install --upgrade pip
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio pygame
RUN pip3 install tensorboard-logger
RUN pip3 install ruamel.base ryd

RUN mkdir /install
WORKDIR /install

RUN pip3 install jsonpickle==0.9.6
# install Sacred (from OxWhirl fork)
RUN pip3 install setuptools
RUN pip3 install sacred

# Install pymongo
RUN pip3 install pymongo

RUN pip3 install gym==0.17.2
RUN pip3 install numpy==1.15.4
RUN pip3 install ray==0.6.6
RUN pip3 install tensorboard==1.14.0
RUN pip3 install tensorboardX==1.6
RUN pip3 install tensorflow-gpu==1.14.0
RUN pip3 install tensorflow-probability==0.7.0
RUN pip3 install opencv-python==4.1.0.25
RUN pip3 install gym-super-mario-bros==7.1.6
RUN pip3 install pybullet==2.5.0
RUN pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
# RUN pip3 install torch==1.5.0

## -- SMAC
ENV smac_ver 1
RUN pip3 install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /pymarl/3rdparty/StarCraftII

WORKDIR /pymarl

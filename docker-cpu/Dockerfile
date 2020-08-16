FROM ubuntu:18.04
# FROM ubuntu:16.04
MAINTAINER Tarun Gupta

# CUDA includes
#ENV CUDA_PATH /usr/local/cuda
#ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
#ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

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
RUN apt-get -y install python3 python3-dev
RUN apt-get -y install python3-pip unzip
RUN pip3 install --upgrade pip
RUN pip3 install scikit-build cmake
RUN pip3 install scipy pyyaml matplotlib imageio pygame tensorboard-logger ruamel.base \
ryd wheel six numpy mock keras_applications keras_preprocessing

RUN mkdir /install
WORKDIR /install

RUN pip3 install jsonpickle==0.9.6 setuptools sacred pymongo gym ray==0.6.6

# Install tensorflow from scratch
RUN apt-get -y install g++ unzip zip openjdk-11-jdk
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.25.3/bazel-0.25.3-installer-linux-x86_64.sh
RUN chmod +x bazel-0.25.3-installer-linux-x86_64.sh && ./bazel-0.25.3-installer-linux-x86_64.sh && rm -f bazel-0.25.3-installer-linux-x86_64.sh
RUN git clone https://github.com/tensorflow/tensorflow.git /tensorflow
WORKDIR /tensorflow
RUN git checkout r1.15 && ./configure
RUN apt-get -y install python python-dev
RUN bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN pip3 install /tmp/tensorflow_pkg/*.whl

RUN pip3 install tensorboardX==1.6 tensorflow-probability==0.7.0 opencv-python==4.1.0.25 \
 gym-super-mario-bros==7.1.6 pybullet==2.5.0
#RUN pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#
##### -------------------------------------------------------------------
##### install pytorch
##### -------------------------------------------------------------------
RUN pip3 install torch==1.5.0
#
### -- SMAC
ENV smac_ver 1
RUN pip3 install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /pymarl/3rdparty/StarCraftII
#
WORKDIR /pymarl

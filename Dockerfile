ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}
LABEL maintainer="esds_algo"

ENV TRT_VERSION 8.5.3.1
# SHELL ["/bin/bash", "-c"]

RUN mkdir -p /workspace

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# Update CUDA signing key
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-key adv --fetch-keys 3bf863cc.pub
RUN apt-get update

RUN apt-get install -y software-properties-common
# Install requried libraries
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    ffmpeg \
    libboost-dev \
    net-tools \
    vim \
    libmysqlclient-dev \
    tree \
    ffmpeg

RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-wheel \
      libtbb2 \
      libtbb-dev \
      libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# Install TensorRT
RUN v="${TRT_VERSION%.*}-1+cuda${CUDA_VERSION%.*}" &&\
    apt-key adv --fetch-keys 3bf863cc.pub &&\
    apt-get update &&\
    apt-get install -y libnvinfer8=${v} libnvonnxparsers8=${v} libnvparsers8=${v} libnvinfer-plugin8=${v} \
        libnvinfer-dev=${v} libnvonnxparsers-dev=${v} libnvparsers-dev=${v} libnvinfer-plugin-dev=${v} \
        python3-libnvinfer=${v}

# Install PyPI packages
RUN pip3 install --upgrade pip -i https://pypi.doubanio.com/simple/ --trusted-host pypi.douban.com
RUN pip3 install setuptools -i https://pypi.doubanio.com/simple/ --trusted-host pypi.douban.com
COPY whl/torch-1.13.1+cu116-cp38-cp38-linux_x86_64.whl /tmp/torch-1.13.1+cu116-cp38-cp38-linux_x86_64.whl
RUN pip3 install /tmp/torch-1.13.1+cu116-cp38-cp38-linux_x86_64.whl -i https://pypi.doubanio.com/simple/ --trusted-host pypi.douban.com
COPY whl/onnx-1.13.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tmp/onnx-1.13.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install /tmp/onnx-1.13.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -i https://pypi.doubanio.com/simple/ --trusted-host pypi.douban.com
COPY whl/onnxruntime-1.14.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tmp/onnxruntime-1.14.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install /tmp/onnxruntime-1.14.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -i https://pypi.doubanio.com/simple/ --trusted-host pypi.douban.com
RUN pip3 install Pillow -i https://pypi.doubanio.com/simple/ --trusted-host pypi.douban.com
RUN pip3 install torchvision==0.14.1 -i https://pypi.doubanio.com/simple/ --trusted-host pypi.douban.com
RUN pip3 install numpy -i https://pypi.doubanio.com/simple/ --trusted-host pypi.douban.com
RUN pip3 install pytest -i https://pypi.doubanio.com/simple/ --trusted-host pypi.douban.com
RUN pip3 install onnx-simplifier -i https://pypi.doubanio.com/simple/ --trusted-host pypi.douban.com
RUN pip3 install nvidia-pyindex
RUN pip3 install onnx-graphsurgeon
# RUN pip3 install jupyter jupyterlab -i https://pypi.doubanio.com/simple/ --trusted-host pypi.douban.com
# Workaround to remove numpy installed with tensorflow
RUN pip3 install --upgrade numpy

# Install Cmake
COPY cmake-3.14.4-Linux-x86_64.sh /tmp/cmake-3.14.4-Linux-x86_64.sh
RUN cd /tmp && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
COPY TensorRT ${TRT_OSSPATH}

# Install cpp library
COPY json-3.11.2 /tmp/json-3.11.2
COPY libwebsockets-4.3.2 /tmp/libwebsockets-4.3.2
COPY websocketpp-master /tmp/websocketpp-master
COPY opencv-4.7.0 /tmp/opencv-4.7.0
RUN cd /tmp/json-3.11.2/build && rm -rf * && cmake .. && make -j $(nproc) && make install && cd ../../ && rm -rf json-3.11.2
RUN cd /tmp/libwebsockets-4.3.2/build && rm -rf * && cmake .. && make -j $(nproc) && make install && cd ../../ && rm -rf libwebsockets-4.3.2
RUN cd /tmp/websocketpp-master/build && rm -rf * && cmake .. && make -j $(nproc) && make install && cd ../../ && rm -rf websocketpp-master

# Install opencv
RUN cd /tmp/opencv-4.7.0/build && rm -rf * && export CUDA_HOME=/usr/local/cuda && export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 && export PATH=${CUDA_HOME}/bin:${PATH} && cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_TESTS=OFF \ 
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D OPENCV_PYTHON3_INSTALL_PATH=/usr/lib/python3/dist-packages \
    -D WITH_QT=OFF \ 
    -D WITH_OPENGL=OFF \
    -D WITH_CUDA=ON \ 
    -D WITH_CUBLAS=1 \
    -D CUDA_GENERATION=Turing \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv-4.7.0/opencv_contrib/modules .. && \
    make -j $(nproc) && \
    make install && cd ../../ && rm -rf opencv-4.7.0

RUN echo /usr/local/lib > /etc/ld.so.conf.d/opencv.conf && ldconfig
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/algo/device/light:/algo/device/plate:/algo/device/power:/algo/device/switch" > /root/.bashrc

COPY algo /algo
# RUN rm /algo/dress/weights
COPY weights /algo/dress/weights

ENV PATH="${PATH}:/usr/local/bin/ngc-cli:${TRT_OSSPATH}/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /algo

CMD ["/bin/bash"]

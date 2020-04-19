#!/bin/bash
set -e

TENSORFLOW_VERSION="v2.1.0"
# This is already done in docker image
# # Install Go (for TF build)
# RUN curl https://dl.google.com/go/go1.13.10.linux-amd64.tar.gz -o go.tar.gz \
# && tar -xvf go.tar.gz \
# && rm -rf go.tar.gz \
# && mv go /usr/local

# ENV GOROOT /usr/local/go
# ENV PATH $GOPATH/bin:$GOROOT/bin:$PATH

# # Install Bazelisk as Bazel
# RUN go get github.com/bazelbuild/bazelisk && ln -s /root/go/bin/bazelisk /usr/local/bin/bazel

poetry run pip install six numpy wheel setuptools mock 
poetry run pip install keras_applications keras_preprocessing  --no-deps

git clone https://github.com/tensorflow/tensorflow.git -b ${TENSORFLOW_VERSION} --depth 1
cd tensorflow/

# Run the configure
# TODO: change CC_OPT_FLAGS to precise build, instead of native
TF_ENABLE_XLA=true \
TF_NEED_CUDA=true \
TF_NEED_TENSORRT=true \
TF_NEED_OPENCL_SYCL=false \
TF_NEED_ROCM=false \
TF_SET_ANDROID_WORKSPACE=false \
TF_CUDA_COMPUTE_CAPABILITIES=6.1 \
TF_CUDA_CLANG=false \
CC_OPT_FLAGS="-march=native -Wno-sign-compare" \
PYTHON_BIN_PATH=/work/.cache/poetry/thc-net-KQLMmzPP-py3.7/bin/python \
USE_DEFAULT_PYTHON_LIB_PATH=1 \
GCC_HOST_COMPILER_PATH=/usr/bin/gcc \
./configure

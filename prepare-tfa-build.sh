#!/bin/bash
set -e

TENSORFLOW_ADDONS_VERSION="v0.10.0"
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

# poetry run pip install six numpy wheel setuptools mock 
# poetry run pip install keras_applications keras_preprocessing  --no-deps

# git clone https://github.com/tensorflow/tensorflow.git -b ${TENSORFLOW_VERSION} --depth 1
# cd tensorflow/

# # Run the configure
# # TODO: change CC_OPT_FLAGS to precise build, instead of native
# TF_ENABLE_XLA=true \
# TF_NEED_CUDA=true \
# TF_NEED_TENSORRT=true \
# TF_NEED_OPENCL_SYCL=false \
# TF_NEED_ROCM=false \
# TF_SET_ANDROID_WORKSPACE=false \
# TF_CUDA_COMPUTE_CAPABILITIES=6.1 \
# TF_CUDA_CLANG=false \
# CC_OPT_FLAGS="-march=native -Wno-sign-compare" \
# PYTHON_BIN_PATH=/work/.cache/poetry/thc-net-KQLMmzPP-py3.7/bin/python \
# USE_DEFAULT_PYTHON_LIB_PATH=1 \
# GCC_HOST_COMPILER_PATH=/usr/bin/gcc \
# ./configure


git clone https://github.com/tensorflow/addons.git -b ${TENSORFLOW_ADDONS_VERSION} --depth 1
cd addons
# export PATH=${PATH}:/usr/sbin:/usr/local/sbin:/sbin

apt update && apt install -y rsync --no-install-recommends
export TF_NEED_CUDA="1"

# Set these if the below defaults are different on your system
export TF_CUDA_VERSION="10.2"
export TF_CUDNN_VERSION="7"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"
export TMP="/tmp"
# This script links project with TensorFlow dependency
python ./configure.py

bazel build --enable_runfiles build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl

#    42  apt install rsync --no-install-recommends
#    43  exit
#    44  . /work/.cache/poetry/thc-net-KQLMmzPP-py3.7/bin/activate
#    45  cd addons/
#    46  bazel-bin/build_pip_pkg artifacts
#    47  rsync
#    48  apt install rsync --no-install-recommends
#    49  rsync
#    50  bazel-bin/build_pip_pkg artifacts
#    51  pip install artifacts/tensorflow_addons-*.whl
#    52  history
# PATH=${PATH}:/sbin
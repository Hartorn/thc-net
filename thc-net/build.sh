#!/bin/bash
set -e

cd tensorflow/

TMP=/tmp \
bazel build \
--local_ram_resources=HOST_RAM*0.5 \
--local_cpu_resources=HOST_CPUS-2 \
//tensorflow/tools/pip_package:build_pip_package \
--config=noaws \
--config=nogcp \
--config=v2 \
--config=mkl \
--config=opt


mkdir -p /work/result_tf
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /work/result_tf
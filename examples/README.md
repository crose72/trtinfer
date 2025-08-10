# Description

Examples of deep learning inference using TensorRT.

# Getting Started

A docker container with TensorRT, CUDA, and OpenCV already installed is provided as the
development environment.

```
cd scripts
./run_dev_ubuntu-22.04.sh
cd examples
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```
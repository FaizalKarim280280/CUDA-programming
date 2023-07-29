# CUDA-programming
This repository contains implementation codes from the course: <a href="https://youtube.com/playlist?list=PL4YhK0pT0ZhXX5i5TlQzSprfJ78kjHV3i">IIT Madras GPU Programming, Spring 2021</a>

## Requirements
1. NVIDIA GPU that supports CUDA.
2. CUDA Toolkit: <a href="https://developer.nvidia.com/cuda-downloads">download CUDA Toolkit</a> \
To check if cuda is installed, run the following command: <code>nvcc --version</code>

## Running the code
1. In the terminal, run the following commands:
```
nvcc <filename>.cu
./a.out
```
2. [Optional] For running the code on IIITH ada cluster:
```
modue load u18/cuda/11.6
sinteractive -c 10 -g 1
nvcc <filename>.cu
./a.out
```

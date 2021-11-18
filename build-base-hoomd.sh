#!/bin/bash

rm -rf .build/hoomd

cmake -B .build/hoomd -S hoomd-blue -DENABLE_GPU=$ENABLE_GPU -DCMAKE_CUDA_ARCHITECTURES=$CUDAARCHS

cmake --build .build/hoomd -j$BUILD_THREADS

cmake --install .build/hoomd
#!/bin/bash

rm -rf .build/plugins

cmake -B .build/plugins -S plugins -DENABLE_HIP=$ENABLE_GPU -DCMAKE_CUDA_ARCHITECTURES=$CUDAARCHS

cmake --build .build/plugins -j$BUILD_THREADS

cmake --install .build/plugins
#!/bin/bash

rm -rf .build/plugins

cmake -B .build/plugins -G Ninja -S plugins -DENABLE_HIP=$ENABLE_GPU -DCMAKE_CUDA_ARCHITECTURES=$CUDAARCHS -DCMAKE_CUDA_FLAGS="--compiler-bindir=$CC" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$CC
#!/bin/bash

rm -rf .build/hoomd

cmake -B .build/hoomd -S hoomd-blue -DENABLE_GPU=$ENABLE_GPU -DBUILD_HPMC=OFF -DBUILD_METAL=OFF -DSINGLE_PRECISION=OFF -DENABLE_TBB=OFF -DCMAKE_CUDA_ARCHITECTURES=$CUDAARCHS -DCMAKE_BUILD_TYPE=Release

cmake --build .build/hoomd -j$BUILD_THREADS

cmake --install .build/hoomd
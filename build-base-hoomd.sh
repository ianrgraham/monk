#!/bin/bash

rm -rf .build/hoomd

export CC=/mnt/rrio1/home/opt/seas/pkg_sl79/gcc-7.3.0/gcc-7.3.0/bin/gcc

cmake -B .build/hoomd -S hoomd-blue -DENABLE_GPU=$ENABLE_GPU -DBUILD_HPMC=OFF -DBUILD_METAL=OFF -DSINGLE_PRECISION=OFF -DENABLE_TBB=OFF -DCMAKE_CUDA_ARCHITECTURES=$CUDAARCHS -DCMAKE_BUILD_TYPE=Release

cmake --build .build/hoomd -j$BUILD_THREADS

cmake --install .build/hoomd
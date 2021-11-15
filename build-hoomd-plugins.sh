#!/bin/bash

cmake -B .build/plugins -S plugins -DENABLE_GPU=$ENABLE_GPU

cmake --build .build/plugins -j$BUILD_THREADS

cmake --install .build/plugins
#!/bin/bash

# will have to not hardcode in the future
NUM_THREADS=6
ENABLE_GPU=ON

cmake -B .build/plugins -S plugins -DENABLE_GPU=$ENABLE_GPU

cmake --build .build/plugins -j$NUM_THREADS

cmake --install .build/plugins
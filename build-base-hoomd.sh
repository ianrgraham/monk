#!/bin/bash

# will have to not hardcode in the future
NUM_THREADS=6
ENABLE_GPU=ON

cmake -B .build/hoomd -S hoomd-blue -DENABLE_GPU=$ENABLE_GPU

cmake --build/hoomd .build -j$NUM_THREADS

cmake --install/build .build
#!/bin/bash

cmake -B .build/hoomd -S hoomd-blue -DENABLE_GPU=$ENABLE_GPU

cmake --build .build/hoomd -j$BUILD_THREADS

cmake --install .build/hoomd
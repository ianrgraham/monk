#!/bin/bash

# $SINGLE_PRECISION is ON, set $HOOMD_LONGREAL_SIZE to 32, else 64
if [ "$SINGLE_PRECISION" = "ON" ]; then
    HOOMD_LONGREAL_SIZE=32
    BUILD_DIR=.build/hoomd
else
    HOOMD_LONGREAL_SIZE=64
    BUILD_DIR=.build/hoomd-double
fi

cmake --build $BUILD_DIR -j$BUILD_THREADS

cmake --install $BUILD_DIR

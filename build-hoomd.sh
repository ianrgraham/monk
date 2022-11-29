#!/bin/bash

cmake --build .build/hoomd -j$BUILD_THREADS

cmake --install .build/hoomd

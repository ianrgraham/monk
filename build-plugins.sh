#!/bin/bash

cmake --build .build/plugins -j$BUILD_THREADS

cmake --install .build/plugins

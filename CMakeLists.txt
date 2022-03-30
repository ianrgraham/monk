CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)

project (MONK LANGUAGES C CXX)

set(ENABLE_GPU ON CACHE BOOL "Enable CUDA GPU" FORCE)

add_subdirectory(hoomd-blue)  # not currently working, use build script
add_subdirectory(plugins)
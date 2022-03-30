CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)

project (MONK LANGUAGES C CXX)

set(ENABLE_GPU ON CACHE BOOL "Enable CUDA GPU" FORCE)

add_subdirectory(hoomd-blue)
# foreach(subdir "plugins")
#   add_subdirectory(${subdir})
# endforeach()
# add_subdirectory(plugins)

# add_dependencies(plugins HOOMD)
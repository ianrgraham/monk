# For installation as an external plugin
CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)

# Name the plugin project
project(plugins LANGUAGES C CXX)

# Find the installed HOOMD
find_package(HOOMD 3.0.0 REQUIRED)

message(STATUS "Found HOOMD ${HOOMD_VERSION}: ${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")

# Force installation to the HOOMD installation location
set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)
set(CMAKE_CXX_STANDARD 20)

# add subdirectories
add_subdirectory(pair_plugin)
# add_subdirectory(forward_flux)
# add_subdirectory(nlist_plugin)
add_subdirectory(neb_plugin)

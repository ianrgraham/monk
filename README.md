# monk
The name here doesn't mean much, but this repo is essentially a collection of ready-to-go scripts and utilities for running simulations using HOOMDv3. Research workflows that depend strictly upon this repository can be found in the `workflows` directory. The workflow manager used here is `Snakemake`.

# Dependencies

 - conda >= 4.10
 - direnv >= 2.25
 - pybind11 >= 2.2
 - Eigen >= 3.2
 - CMake >= 3.9

# Setup
We use `direnv` to automatically configure the environment of the project. With it installed, cd'ing into the project root will create/activate a local python environment that we will use to build HOOMD and run our analyses.

To build HOOMD and any plugins, run the included build scripts

```
./build-base-hoomd.sh
./build-hoomd-plugins.sh
```

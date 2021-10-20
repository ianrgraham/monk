# monk
The name here doesn't mean much, but this repo is essentially a collection of ready-to-go scripts and utilities for running simulations using HOOMD v3. 

# Setup
Since HOOMD is a bit stuborn with its insistence on only distributing its binaries though `conda`, that's how we'll setup our environment. `environment.yml` can be used to create a new conda env with the right dependencies. If you have `direnv` installed with the proper extensions for `anaconda`, the environment will be automatically loaded by the shell.
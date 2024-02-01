# cuDisc

GPU accelerated protoplanetary disc model

To compile correctly, the line 

    CUDA_HOME = /usr/local/cuda-12.0

in the makefile must be set to the correct location for your machine's CUDA installation.

Tests from Robinson et al. 2024 can be run by running the python scripts in the /tests/scripts directory.

/codes contains two example simulations, a 1D run and a 2D run. In codes/python, an ipython notebook contains examples of how to plot the results from these example simulations.
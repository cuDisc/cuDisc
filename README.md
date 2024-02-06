# cuDisc: a GPU accelerated protoplanetary disc model

cuDisc is a code that allows the study of axisymmetric protoplanetary discs in both the radial and vertical dimensions. Dust dynamics, growth and fragmentation, and 2D radiative transfer are all included, alongside a 1D gas evolution model with 2D gas structure calculated through hydrostatic equilibrium. For more details, see Robinson et al. 2024. If you use cuDisc in any of your research, this is also the paper we wish you to cite. 

### Usage ###

To get the code, clone this repository to your machine. To compile the source files successfully, the line 

    CUDA_HOME = /usr/local/cuda-12.0

in the makefile must be set to the correct location for your machine's CUDA installation.

The source files can then be compiled by making one of the example tests using

    make test_*

replacing * with the desired test (e.g. adv_diff), or by making one of the example simulations (e.g. steadyTD) using

    make steadyTD

You can also build the code as a static library with the command

    make lib

There is a second makefile in the codes/ directory that shows an example of how to use the static library for your own simulation files.

Tests from Robinson et al. 2024 can be run by running the python scripts in the tests/scripts/ directory.

codes/ contains two example simulations, a 1D run (1Ddisc.cpp) and a 2D run (steadyTD.cpp). In codes/python/, an ipython notebook contains examples of how to plot the results from these example simulations.

### Authors ###

- [Richard Booth](https://github.com/rbooth200)
- [Alfie Robinson](https://github.com/alfrob98)

### Contact ###

If you have any questions or discover any issues, feel free to email [Alfie](mailto:a.robinson21@imperial.ac.uk) or [Richard](r.a.booth@leeds.ac.uk). 
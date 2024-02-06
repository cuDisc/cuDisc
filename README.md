# cuDisc: a GPU accelerated protoplanetary disc model

cuDisc is designed for studying the evolution protoplanetary discs in both the radial and vertical dimensions, assuming axisymmetry. The processes included in cuDisc are 2D dust advection-diffusion, dust coagulation/fragmentation, and radiative transfer. A 1D evolution model is also included, with the 2D gas structure calculated via vertical hydrostatic equilibrium. For more details, see Robinson et al. (2024). If you use cuDisc in any of your research, this is also the paper we wish you to cite. 

### Usage ###

You will need a NVIDIA GPU to use cuDisc.

To get the code, clone this repository to your machine or download one of the releases. To compile the source files successfully, the line 

    CUDA_HOME = /usr/local/cuda-12.0

in the makefile must be set to the correct location for your machine's CUDA installation.

The source files can then be compiled by making one of the example tests using

    make test_*

replacing * with the desired test (e.g. adv_diff). Currently the following tests are included: 
* `test_adv_diff`: A basic dust advection-diffusion test.
* `test_coag_const`: A basic coagulation test with the constant kernel.
* `test_coagdustpy`: A comparison of cuDisc's dust growth and fragmentation routines against DustPy's for 1D vertically integrated disc (without radial drift).
* `test_pinte_graindist` : A 2D radiative-transfer test in the spirit of the Pinte et al. (2009) benchmark.

The example simulations can be compiled via:
    
    make steadyTD

or

    make 1Ddisc

These examples are contained in  `codes/` directory, and includes both a 1D run (`1Ddisc.cpp`) and a 2D run (`steadyTD.cpp`). The 1D run is a gas + two-population dust model (following Birnstiel et al. 2012) with photoevaporation. The 2D run is similar to the 2D-coagulation, fragmentation and radiative transfer model included presented in the code paper.
An ipython notebook (In `codes/python/disc_plotter.ipynb`) contains examples of how to plot the results from these example simulations.

You can also build the code as a static library with the command

    make lib

There is a second makefile in the `codes/` directory that shows an example of how to use the static library for your own simulation files.

The tests in Robinson et al. 2024 can be run via the python scripts in the `tests/scripts/` directory.


### Attribution ###

If you use this code in your research please cite the code paper, Robinson et al. (2024): [MNRAS]() [ADS]() [arXiv](). 
For a list of works using cuDisc, see the citations [here]().

### Authors ###

- [Richard Booth](https://github.com/rbooth200)
- [Alfie Robinson](https://github.com/alfrob98)

### Contact ###

If you have any questions or discover any issues, please open an issue on [github](https://github.com/cuDisc/cuDisc/issues). Alternatively, feel free to email us: [Alfie](mailto:a.robinson21@imperial.ac.uk), [Richard](mailto:r.a.booth@leeds.ac.uk). 
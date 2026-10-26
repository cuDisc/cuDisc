.. _quickstart:

Quickstart
==========

This page walks you through running your first cuDisc simulation — the 2D
steady-state transition disc example — and plotting the results. It assumes
you have already :ref:`installed <installation>` the code and compiled the
example targets.

Running the 2D example
-----------------------

Compile and run the 2D transition disc simulation:

.. code-block:: bash

   make steadyTD
   ./steadyTD

The simulation couples dust dynamics, grain-size coagulation/fragmentation,
and radiative transfer in a gas disc with a cavity. Output snapshots are
written to ``codes/outputs/steadyTD/`` as the run progresses. Typical
simulation times range from minutes (short evolutionary times) to hours
(Myr-scale evolution), depending on GPU speed and grid resolution.

You will see the time-step counter printed to the terminal as the run
progresses:

.. code-block:: text

   Initial gas mass: 0.0127 M_sun
   Initial dust mass: 0.000127 M_sun
   Computing initial temperature structure
   Iteration: 0
   ...
   t = 1000.0 years
   dt = 23.4 years
   Years per second: 450.2

The simulation writes data at the times specified in ``codes/outputs/steadyTD/2Dtimes.txt``.

Plotting the results
--------------------

Open the analysis notebook:

.. code-block:: bash

   cd codes/python
   jupyter notebook disc_plotter.ipynb

The notebook demonstrates how to read the binary output files and make
standard disc plots such as the dust surface density map, the grain-size
distribution at a given radius, and the temperature structure.

Running the 1D example
-----------------------

A simpler starting point is the 1D gas-plus-dust model, which runs in seconds:

.. code-block:: bash

   make 1Ddisc
   ./1Ddisc

This evolves a 1D disc with a two-population dust model (following
Birnstiel et al. 2012) and photoevaporative wind losses. Output is written
to ``codes/outputs/1Ddisc/``.

Understanding the simulation files
------------------------------------

Each example simulation is a single self-contained C++ source file in the
``codes/`` directory (e.g. ``codes/steadyTD.cpp``). The typical structure is:

1. **Grid setup** — create a :cpp:class:`Grid` object with the desired number
   of cells and radial/angular extents.
2. **Size grid setup** — create a :cpp:class:`SizeGrid` specifying the grain
   size range and number of size bins.
3. **Opacity setup** — load a :cpp:class:`DSHARP_opacs` opacity table and
   interpolate it onto the grain-size and wavelength grids.
4. **Field initialisation** — allocate :cpp:class:`Field` and
   :cpp:class:`Field3D` objects for gas, dust, temperature, and radiative
   flux; set initial conditions.
5. **Solver construction** — create :cpp:class:`DustDynamics`,
   :cpp:class:`FLD_Solver`, and coagulation integrator objects.
6. **Time-stepping loop** — advance the simulation by calling the solvers in
   sequence and writing output at the desired times.

The :doc:`user_guide/index` explains each of these steps in detail.

Next steps
----------

- :doc:`user_guide/index` — how to build your own simulation from scratch.
- :doc:`examples/index` — detailed walkthroughs of each example.
- :doc:`physics/index` — the equations and algorithms behind each module.

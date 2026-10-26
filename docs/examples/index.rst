.. _examples:

Examples
========

cuDisc ships with three example simulations that cover its main modes of
operation. Each is a single C++ source file in the ``codes/`` directory.

.. toctree::
   :maxdepth: 1

   steadyTD
   isoPD
   1Ddisc

.. _example_steadyTD:

2D Transition Disc (``steadyTD``)
----------------------------------

**Source file:** ``codes/steadyTD.cpp``

**Physics:** 2D dust dynamics + coagulation/fragmentation + multifrequency
radiative transfer in a gas disc with a central cavity.

This is the flagship example, corresponding to the fiducial run presented in
Robinson et al. (2024). A gas disc with a cavity is initialised with a
population of dust grains spanning five orders of magnitude in grain size.
The code evolves the grain-size distribution self-consistently with the dust
dynamics and the disc temperature over 1 Myr.

Key parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Value
     - Description
   * - ``NR``
     - 150
     - Number of radial cells
   * - ``Nphi``
     - 150
     - Number of polar-angle cells
   * - ``Rmin``
     - 3 au
     - Inner radial boundary
   * - ``Rmax``
     - 20 au
     - Outer radial boundary
   * - ``theta_max``
     - π/4
     - Maximum polar angle from midplane
   * - ``n_spec``
     - 135
     - Number of grain-size bins
   * - ``alpha``
     - 10\ :sup:`-3`
     - Turbulence parameter
   * - ``v_frag``
     - 10 m s\ :sup:`-1`
     - Fragmentation velocity

Build and run:

.. code-block:: bash

   make steadyTD
   ./steadyTD

Output is written to ``codes/outputs/steadyTD/``.

.. _example_isoPD:

2D Isothermal Disc (``isoPD``)
--------------------------------

**Source file:** ``codes/isoPD.cpp``

**Physics:** 2D dust dynamics + coagulation/fragmentation with a fixed,
vertically isothermal temperature structure.

This example is identical to the transition disc run but omits the
temperature solver. The temperature is prescribed as a power law in radius,
making it substantially cheaper computationally. It is useful for
parameter studies in which the temperature is not expected to change
significantly.

Build and run:

.. code-block:: bash

   make isoPD
   ./isoPD

Output is written to ``codes/outputs/isoPD/``.

.. _example_1Ddisc:

1D Disc (``1Ddisc``)
----------------------

**Source file:** ``codes/1Ddisc.cpp``

**Physics:** 1D viscous gas evolution + two-population dust model
(Birnstiel et al. 2012) + photoevaporative wind (Picogna et al. 2019).

This example runs in the 1D approximation, collapsing the vertical
dimension. The gas evolves under viscous diffusion and a parametric
photoevaporative wind. The dust is tracked with a two-population model that
approximates the full size distribution with a large and a small
representative grain. This is the most computationally inexpensive option
and is useful for long evolution timescales.

Build and run:

.. code-block:: bash

   make 1Ddisc
   ./1Ddisc

Output is written to ``codes/outputs/1Ddisc/``.

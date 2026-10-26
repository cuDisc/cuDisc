.. _examples_isoPD:

isoPD — isothermal disc
========================

``isoPD`` evolves a 2D protoplanetary disc with a fixed isothermal
temperature profile. Radiative transfer is disabled, making this a much
cheaper run useful for testing dust dynamics and coagulation in isolation,
and for quick parameter sweeps.

Source file: ``codes/isoPD.cpp``

Physical setup
--------------

The temperature profile is a power law

.. math::

   T(R) = T_0 \left(\frac{R}{R_0}\right)^{-q}

with *T*\ :sub:`0` = 150 K at *R*\ :sub:`0` = 1 au and *q* = 0.5. The
disc aspect ratio *h*/*R* = *c*\ :sub:`s`/(*R* Ω\ :sub:`K`) is therefore
also a power law, giving a flared disc.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Value
   * - Stellar mass
     - 1 M\ :sub:`☉`
   * - Disc inner radius
     - 1 au
   * - Disc outer radius
     - 100 au
   * - Radial cells (*NR*)
     - 200 (log-spaced)
   * - Polar cells (*Nphi*)
     - 100 (subdiv-spaced)
   * - Grain-size bins (*n_spec*)
     - 100
   * - Turbulence α
     - 1 × 10\ :sup:`−3`

How to run
----------

.. code-block:: bash

   make isoPD
   mkdir -p output/isoPD
   ./isoPD

Visualisation
-------------

Use ``codes/disc_plotter.ipynb`` with the ``isoPD`` output directory. The
notebook can display:

- Dust surface density as a 2D map.
- Radial dust-to-gas ratio profiles.
- The grain-size distribution at selected radii.

When to use isoPD
-----------------

``isoPD`` is appropriate when:

- You want to study dust dynamics or coagulation without the overhead of
  radiative transfer.
- You are testing a new disc model and want rapid turnaround.
- You are running a parameter survey over α, *v*\ :sub:`frag`, or grain
  properties.

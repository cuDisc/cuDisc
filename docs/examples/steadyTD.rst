.. _examples_steadyTD:

steadyTD — 2D disc with coagulation and FLD
============================================

``steadyTD`` evolves a 2D protoplanetary disc toward a quasi-steady state
with self-consistent dust dynamics, grain-size coagulation, and
multifrequency radiative transfer. It is the primary demonstration
example of cuDisc.

Source file: ``codes/steadyTD.cpp``

Physical setup
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Value
   * - Stellar mass
     - 1 M\ :sub:`☉`
   * - Stellar luminosity
     - 1 L\ :sub:`☉`
   * - Stellar temperature
     - 4000 K
   * - Disc inner radius
     - 3 au
   * - Disc outer radius
     - 20 au
   * - Radial cells (*NR*)
     - 150 (log-spaced)
   * - Polar cells (*Nphi*)
     - 150 (subdiv-spaced)
   * - Grain-size bins (*n_spec*)
     - 135
   * - Grain-size range
     - 0.1 μm – 10 cm
   * - Turbulence α
     - 1 × 10\ :sup:`−3`
   * - Fragmentation velocity
     - 10 m/s
   * - Gas surface density
     - Σ ∝ *R*\ :sup:`−1` normalised to 20 g/cm² at 1 au

How to run
----------

.. code-block:: bash

   # Compile (once)
   make steadyTD

   # Create output directory and run
   mkdir -p output/steadyTD
   ./steadyTD

Output is written to ``output/steadyTD/`` in the binary format described
in :ref:`user_guide_output`.

Visualisation
-------------

Open ``codes/disc_plotter.ipynb`` and set the output path to
``output/steadyTD``. The notebook produces:

- Dust surface density Σ\ :sub:`d` (R, grain size) as a 2D colour map.
- Midplane temperature *T*(*R*).
- Vertically integrated grain-size distribution.

Expected results
----------------

After approximately 1 Myr of disc evolution (wall-clock time of a few
hours on an A100 GPU), the disc reaches a quasi-steady state with:

- A fragmentation barrier limiting the maximum grain size at small radii.
- A drift barrier at larger radii.
- A warm atmosphere heated by stellar irradiation.

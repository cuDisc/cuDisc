.. _user_guide:

User Guide
==========

This guide explains how to write a custom cuDisc simulation. It follows the
structure of the example simulations in ``codes/`` and covers each step in
order. For the physical motivation behind each component, refer to
:doc:`../physics/index`.

.. toctree::
   :maxdepth: 1

   grid
   fields
   gas
   dust
   opacity
   temperature
   coagulation
   timestepping
   output

Overview
--------

A cuDisc simulation is a C++ program that links against the cuDisc static
library (``lib/libcudisc.a``). The standard workflow is:

1. **Create a** :cpp:class:`Grid` — defines the computational mesh in
   (R, θ) space.
2. **Create a** :cpp:class:`SizeGrid` — defines the logarithmic grain-size
   bins.
3. **Load opacities** — read and interpolate a :cpp:class:`DSHARP_opacs`
   table.
4. **Allocate fields** — create :cpp:class:`Field` and
   :cpp:class:`Field3D` objects for all physical quantities.
5. **Set initial conditions** — fill the fields with the desired starting
   state.
6. **Construct solvers** — instantiate :cpp:class:`DustDynamics`,
   :cpp:class:`FLD_Solver`, and a coagulation integrator.
7. **Run the time loop** — advance the simulation, calling each solver in
   operator-splitting order, and write output.

Headers are in ``headers/`` and sources in ``src/``. All public-facing types
live in the ``headers/`` directory; there is no separate namespace — names
are accessed directly.

Typical include block
~~~~~~~~~~~~~~~~~~~~~

Most simulations include:

.. code-block:: cpp

   #include "grid.h"           // Grid, GridRef, create_field, create_field3D
   #include "field.h"          // Field<T>, Field3D<T>
   #include "cuda_array.h"     // CudaArray<T>
   #include "constants.h"      // au, Msun, year, GMsun, sigma_SB, ...
   #include "dustdynamics.h"   // DustDynamics, Prims, BoundaryFlags
   #include "sources.h"        // Sources (drag, settling source terms)
   #include "gas1d.h"          // 1D gas evolution
   #include "hydrostatic.h"    // compute_hydrostatic_equilibrium
   #include "stellar_irradiation.h"  // Star, compute_stellar_heating_with_scattering
   #include "DSHARP_opacs.h"   // DSHARP_opacs, SizeGrid
   #include "FLD.h"            // FLD_Solver, WavelengthBinner
   #include "bins.h"           // WavelengthBinner
   #include "file_io.h"        // write_file, write_grids, read_restart_file
   #include "coagulation/coagulation.h"   // CoagulationRate
   #include "coagulation/integration.h"   // BS32Integration
   #include "coagulation/fragments.h"     // SimpleErosion, BirnstielKernel

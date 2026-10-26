.. _user_guide_grid:

The Grid
========

cuDisc uses a 2D poloidal mesh in cylindrical radius *R* and polar angle
*θ* measured from the midplane. Cells are bounded by surfaces of constant
*R* and constant *θ*, so the cell faces in the *Z*-direction are not
horizontal — they are cones, not planes.

.. note::

   The convention here is *θ = 0* at the midplane and *θ* increasing toward
   the pole. This differs from the usual astronomical convention where
   co-latitude is measured from the pole. The grid therefore spans
   *0 ≤ θ ≤ θ*:sub:`max`, with the midplane at the lower boundary.

Creating a Grid
---------------

A :cpp:class:`Grid` is constructed from a :cpp:struct:`Grid::params` struct:

.. code-block:: cpp

   Grid::params p;
   p.NR    = 150;          // number of radial cells (excluding ghost cells)
   p.Nphi  = 150;          // number of polar cells
   p.Nghost = 2;           // number of ghost-cell layers (must be >= 2)

   p.Rmin      = 3. * au;      // inner radial boundary (cm)
   p.Rmax      = 20. * au;     // outer radial boundary (cm)
   p.theta_min = 0.;           // lower polar-angle boundary (midplane)
   p.theta_max = M_PI / 4.;   // upper polar-angle boundary

   p.R_spacing     = RadialSpacing::log;             // logarithmic in R
   p.theta_spacing = ThetaSpacing::subdiv;           // subdivided polar grid
   p.theta_subdiv  = M_PI / 9.;                      // subdivision angle
   p.theta_power   = 0.75;                           // power for outer region

   Grid g(p);

Grid spacing options
~~~~~~~~~~~~~~~~~~~~

Radial spacing is controlled by ``p.R_spacing``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``RadialSpacing::linear``
     - Uniform spacing in *R*.
   * - ``RadialSpacing::log``
     - Logarithmic spacing in *R*. **Recommended** for most disc simulations.
   * - ``RadialSpacing::power``
     - Power-law spacing :math:`\Delta R \propto R^{p_R}`, set via
       ``p.R_power``.

Polar-angle spacing is controlled by ``p.theta_spacing``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``ThetaSpacing::linear``
     - Uniform spacing in *θ*.
   * - ``ThetaSpacing::subdiv``
     - Fine uniform spacing from *θ*:sub:`min` to ``theta_subdiv``, then
       power-law spacing from ``theta_subdiv`` to *θ*:sub:`max`. This
       concentrates cells near the midplane while still reaching large
       heights.
   * - ``ThetaSpacing::power``
     - Power-law spacing :math:`\Delta\theta \propto \theta^{p_\theta}` set
       via ``p.theta_power``.

Accessing grid quantities
-------------------------

The :cpp:class:`Grid` object provides cell-centre and cell-edge positions in
both cylindrical (*R*, *Z*) and spherical (*r*, *θ*) coordinates, as well
as cell volumes and face areas:

.. code-block:: cpp

   // Cylindrical radius at cell centre (i) and edge (i)
   double Rc = g.Rc(i);   // cell centre
   double Re = g.Re(i);   // cell edge

   // Height above midplane at cell (i, j)
   double Zc = g.Zc(i, j);   // Z at cell centre
   double Ze = g.Ze(i, j);   // Z at lower face (j edge)

   // Spherical radius at cell (i, j)
   double rc = g.rc(i, j);

   // Geometry factors
   double V   = g.volume(i, j);   // cell volume
   double AR  = g.area_R(i, j);   // face area in the R-direction
   double AZ  = g.area_Z(i, j);   // face area in the Z-direction

Ghost cells
-----------

Ghost cells extend the active domain by ``Nghost`` layers on each side.
Physical cells run from index ``g.Nghost`` to ``g.NR + g.Nghost - 1`` in
the radial direction, and from ``g.Nghost`` to ``g.Nphi + g.Nghost - 1``
in the polar direction. Ghost cells are used by the solvers to implement
boundary conditions.

.. warning::

   Always use ``g.Nghost`` as the loop start index when iterating over
   *physical* cells, and ``g.NR + g.Nghost`` (or ``g.Nphi + g.Nghost``) as
   the loop end index. Iterating from 0 to ``g.NR`` without adding
   ``g.Nghost`` will skip the ghost cells inconsistently.

Multiple grids and sub-grids
-----------------------------

:cpp:class:`GridManager` supports adding sub-grids that cover a subset of
the radial range. This is useful for regions that need higher resolution
than the base grid. Sub-grids are created with:

.. code-block:: cpp

   GridManager gm(g_main);
   Grid g_sub = gm.add_subgrid(R_in, R_out);

Data can be copied between the main grid and a sub-grid using
``gm.copy_to_subgrid`` and ``gm.copy_from_subgrid``.

Physical constants
------------------

Physical and astronomical constants are defined in ``headers/constants.h``
and are available without any prefix:

.. code-block:: cpp

   au        // astronomical unit in cm
   Msun      // solar mass in g
   Rsun      // solar radius in cm
   year      // Julian year in seconds
   GMsun     // G * Msun in cm^3 s^-2
   sigma_SB  // Stefan-Boltzmann constant
   k_B       // Boltzmann constant
   m_H       // hydrogen atom mass
   R_gas     // gas constant

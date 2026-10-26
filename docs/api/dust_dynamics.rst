.. _api_dust_dynamics:

Dust Dynamics
=============

Defined in ``headers/dustdyn.h``.

SizeGrid
--------

Defines the logarithmic grain-size grid shared by the dust dynamics and
coagulation modules.

.. code-block:: cpp

   class SizeGrid;

SizeGrid constructor
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   SizeGrid(const Grid& g,
            double a_min,
            double a_max,
            int    n_spec,
            double rho_grain);

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``g``
     - The computational grid.
   * - ``a_min``
     - Minimum grain radius [cm].
   * - ``a_max``
     - Maximum grain radius [cm].
   * - ``n_spec``
     - Number of grain-size bins (logarithmically spaced).
   * - ``rho_grain``
     - Grain material density [g/cm³].

Accessors
~~~~~~~~~

.. code-block:: cpp

   int    n_spec()       const;   // Number of bins
   double grain_size(int k) const;// Grain radius of bin k [cm]
   double grain_mass(int k) const;// Grain mass of bin k [g]
   double min_mass()     const;   // Mass of smallest bin [g]

Sources
-------

Collects the per-cell source terms (stellar gravity, gas pressure gradient)
passed to ``DustDynamics``.

.. code-block:: cpp

   class Sources;

Sources constructor
~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   Sources(const Field<double>& T,
           const Field<Prims>&  Ws_g,
           const SizeGrid&      sizes,
           double               floor,
           double               M_star,
           const Field<double>& mu);

DustDynamics
------------

Advances the dust primitive-variable field ``Ws_d`` by one time step using
the second-order finite-volume scheme.

.. code-block:: cpp

   class DustDynamics;

DustDynamics constructor
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   DustDynamics(const Field<double>& D,
                const Field<double>& cs,
                const Sources&       sources,
                double               C_adv,
                double               C_diff,
                double               floor,
                double               gas_floor);

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``D``
     - Turbulent diffusion coefficient field.
   * - ``cs``
     - Isothermal sound-speed field.
   * - ``sources``
     - Source-term object (gravity, pressure gradient).
   * - ``C_adv``
     - Advective CFL coefficient (default 0.4).
   * - ``C_diff``
     - Diffusive CFL coefficient (default 0.2).
   * - ``floor``
     - Dust density floor [g/cm³].
   * - ``gas_floor``
     - Gas density floor [g/cm³].

Advance one time step
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   void operator()(const Grid&       g,
                   Field3D<Prims>&   Ws_d,
                   const Field<Prims>& Ws_g,
                   double            dt);

CFL limit
~~~~~~~~~

.. code-block:: cpp

   double get_CFL_limit(const Grid&         g,
                        const Field3D<Prims>& Ws_d,
                        const Field<Prims>& Ws_g) const;

Returns the maximum time step allowed by the CFL conditions across all cells
and all dust species.

Floor utilities
~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Set all dust densities below `floor` to `floor`
   void floor_above(Field3D<Prims>& Ws_d, double floor);

   // Set cells below the gas density floor height to the floor density
   void compute_gas_floor_height(const Grid& g,
                                 const Field<Prims>& Ws_g,
                                 Field3D<Prims>& Ws_d);

Restart
~~~~~~~

.. code-block:: cpp

   // Read dust fields from a binary restart file
   void read_active(const std::string& dir, int snap,
                    const Grid& g, Field3D<Prims>& Ws_d);

.. _api_radiative_transfer:

Radiative Transfer
==================

Defined in ``headers/FLD.h``, ``headers/star.h``, and ``headers/opacities.h``.

Star
----

Holds the stellar parameters and computes the stellar flux spectrum.

.. code-block:: cpp

   class Star;

Star constructor
~~~~~~~~~~~~~~~~

.. code-block:: cpp

   Star(double GM, double L_star, double T_star);

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``GM``
     - Gravitational parameter *G M*\ :sub:`★` [cm³/s²] (use ``GMsun * M_star``).
   * - ``L_star``
     - Stellar luminosity [erg/s] (use ``Lsun * L_star``).
   * - ``T_star``
     - Stellar effective temperature [K].

Star methods
~~~~~~~~~~~~

.. code-block:: cpp

   // Set the wavelength grid for the stellar spectrum
   void set_wavelengths(int n_lam, const double* lam);

   // Fill stellar flux array from a Planck blackbody spectrum
   void set_blackbody_fluxes();

DSHARP_opacs
------------

Loads and interpolates the DSHARP opacity table (Birnstiel et al. 2018).

.. code-block:: cpp

   class DSHARP_opacs;

Constructors
~~~~~~~~~~~~

.. code-block:: cpp

   // Load opacity table from file
   DSHARP_opacs(const std::string& filename, bool use_stokes);

   // Allocate interpolated opacity arrays for n_spec grain sizes
   DSHARP_opacs(int n_spec, int n_lam);

DSHARP_opacs methods
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Fill wavelength grid (log-spaced from lam_min to lam_max [micron])
   void generate_lam(double lam_min, double lam_max);

   // Fill grain-size grid from a SizeGrid
   void generate_a(const SizeGrid& sizes);

   // Bilinearly interpolate opacity table to the generated a/lam grids
   void interpolate_opacs(const DSHARP_opacs& table);

   // Set the minimum gas opacity contribution (floor)
   void set_k_g_min_grain(double kappa_min);

   // Compute total opacity field: rho*kappa summed over all dust species
   void calculate_total_rhokappa(const Grid& g,
                                 const Field3D<Prims>& Ws_d,
                                 Field<double>& rhokappa_abs,
                                 Field<double>& rhokappa_sca) const;

   // Wavelength array accessor
   const double* lam() const;

WavelengthBinner
----------------

Groups the high-resolution wavelength grid into ``n_bands`` frequency bands
for the multifrequency FLD solver.

.. code-block:: cpp

   class WavelengthBinner;

WavelengthBinner constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   WavelengthBinner(int n_lam, const double* lam, int n_bands);

WavelengthBinner methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Average opacity over each band weighted by the Planck function at T
   void bin_planck(double T, const DSHARP_opacs& opacs,
                   CudaArray<double>& kappa_band) const;

   // Use the opacity at the central wavelength of each band
   void bin_central(const DSHARP_opacs& opacs,
                    CudaArray<double>& kappa_band) const;

   // Write the band wavelength grid to file
   void write_grids(const std::string& dir) const;

FLD_Solver
----------

Solves the multifrequency flux-limited diffusion (FLD) equation for the
radiation energy density *J* and updates the temperature *T*.

.. code-block:: cpp

   class FLD_Solver;

FLD_Solver constructor
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   FLD_Solver(int max_iter, double tol, int n_pcg);

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``max_iter``
     - Maximum outer (temperature) iterations (default 10).
   * - ``tol``
     - Convergence tolerance on the fractional temperature change (default 1e-5).
   * - ``n_pcg``
     - Maximum PCG (inner linear solver) iterations (default 5000).

FLD_Solver methods
~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Solve the FLD equation for all bands and update T
   void solve_multi_band(const Grid& g,
                         double dt,
                         const Field<double>& Cv,
                         const WavelengthBinner& bins,
                         const DSHARP_opacs& opacs,
                         const Field<double>& heat,   // stellar + viscous heating
                         Field<double>& T,
                         Field<double>& J);

   // Set the PCG preconditioner level (0 = diagonal, 1 = ILU0)
   void set_precond_level(int level);

Stellar heating
~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Compute stellar irradiation with single-scattering via ray-tracing
   void compute_stellar_heating_with_scattering(
       const Grid& g,
       const Star& star,
       const DSHARP_opacs& opacs,
       const WavelengthBinner& bins,
       Field<double>& heat);

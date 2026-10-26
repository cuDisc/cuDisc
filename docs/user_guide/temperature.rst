.. _user_guide_temperature:

Temperature and Radiative Transfer
====================================

cuDisc computes the disc temperature self-consistently using a hybrid
method: stellar irradiation is treated with frequency-resolved ray-tracing,
and thermal re-emission is handled with a multifrequency flux-limited
diffusion (FLD) solver.

Creating the FLD solver
------------------------

.. code-block:: cpp

   int    max_iter  = 10;     // maximum FLD iterations per call
   double tolerance = 1.e-5;  // convergence tolerance on the radiative energy
   int    n_pcg     = 5000;   // maximum preconditioned CG iterations

   FLD_Solver FLD(max_iter, tolerance, n_pcg);

Set boundary conditions for the radiative energy:

.. code-block:: cpp

   FLD.set_boundaries(BoundaryFlags::open_R_inner  |
                      BoundaryFlags::open_R_outer  |
                      BoundaryFlags::open_Z_outer);

The heat and J fields
----------------------

Two additional fields are needed for the temperature solver:

.. code-block:: cpp

   Field<double>  heat = create_field<double>(g);      // stellar heating rate (erg cm^-3 s^-1)
   Field3D<double> J   = create_field3D<double>(g, n_bands); // multiband mean intensity

Initialise the radiative energy field with an estimate based on the stellar
heating:

.. code-block:: cpp

   for (int i = 0; i < g.NR + 2*g.Nghost; i++)
       for (int j = 0; j < g.Nphi + 2*g.Nghost; j++)
           for (int k = 0; k < n_bands; k++)
               J[J.index(i,j,k)] = heat[heat.index(i,j)] / n_bands;

Computing stellar heating
--------------------------

Stellar heating includes both direct irradiation and single-scattered light:

.. code-block:: cpp

   Field3D<double> scattering        = create_field3D<double>(g, num_wavelengths);
   Field3D<double> binned_scattering = create_field3D<double>(g, n_bands);

   compute_stellar_heating_with_scattering(star, g, rhok_abs, rhok_sca,
                                           heat, scattering);
   binned_scattering = bins.bin_field(g, scattering, bins.SUM);

Solving for the temperature
----------------------------

With the opacity and heating fields ready, call the FLD solver:

.. code-block:: cpp

   double Cv = 2.5 * R_gas / mu;   // heat capacity per unit mass

   FLD.solve_multi_band(g, dt, Cv, rhok_abs_binned, rhok_sca_binned,
                        rho_tot, heat, binned_scattering, bins.edges, T, J);

On the first call ``dt = 0`` requests a time-independent (steady-state) solution.
For subsequent calls during time evolution, pass the physical time interval.

Iterating to convergence
~~~~~~~~~~~~~~~~~~~~~~~~~

During the initial setup, the temperature must be iterated to convergence
because the opacity depends on the temperature through the grain-size
distribution and the dust density structure (which depends on hydrostatic
equilibrium). A typical initial iteration loop is:

.. code-block:: cpp

   double tol = 1.;
   int    n   = 0;

   while (n < 50 && tol > 1.e-5) {
       Field<double> T_old = create_field<double>(g);
       copy_field(g, T, T_old);

       calculate_total_rhokappa(g, Ws_d, Ws_g, rho_tot, opacs, rhok_abs, rhok_sca);
       rhok_abs_binned = bins.bin_planck(g, rhok_abs, T);
       bin_central(g, rhok_sca, rhok_sca_binned, num_wavelengths, n_bands);
       compute_stellar_heating_with_scattering(star, g, rhok_abs, rhok_sca,
                                               heat, scattering);
       binned_scattering = bins.bin_field(g, scattering, bins.SUM);

       FLD.solve_multi_band(g, 0., Cv, rhok_abs_binned, rhok_sca_binned,
                            rho_tot, heat, binned_scattering, bins.edges, T, J);

       compute_cs2(g, T, cs2, mu);
       compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g);

       tol = fracerr(g, T_old, T);   // max fractional temperature change
       n++;
   }

Preconditioning
~~~~~~~~~~~~~~~~

For poorly conditioned systems (e.g. very optically thick regions), the
solver's preconditioner level can be raised:

.. code-block:: cpp

   FLD.set_precond_level(1);   // stronger preconditioning (slower but more robust)
   FLD.set_precond_level(0);   // default (ILU(0))

Updating temperature in the time loop
--------------------------------------

During the main time loop, the temperature solver is called at intervals
determined by the thermal timescale. A typical strategy calls it once the
fractional change in the physical time exceeds 10% of the time over which
temperature changed by 1%:

.. code-block:: cpp

   if ((t + dt) - t_temp > 0.1 * dt_1perc || t_temp_max_exceeded) {
       calculate_total_rhokappa(g, Ws_d, Ws_g, rho_tot, opacs, rhok_abs, rhok_sca);
       rhok_abs_binned = bins.bin_planck(g, rhok_abs, T);
       bin_central(g, rhok_sca, rhok_sca_binned, num_wavelengths, n_bands);
       compute_stellar_heating_with_scattering(star, g, rhok_abs, rhok_sca,
                                               heat, scattering);
       binned_scattering = bins.bin_field(g, scattering, bins.SUM);

       FLD.solve_multi_band(g, (t + dt) - t_temp, Cv, rhok_abs_binned,
                            rhok_sca_binned, rho_tot, heat, binned_scattering,
                            bins.edges, T, J);
       t_temp = t + dt;

       // Update sound speed and hydrostatic structure
       compute_cs2(g, T, cs2, mu);
       compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, Ws_d, gas_floor, floor);
       compute_D(g, D, Ws_g, cs2, M_star, alpha, Sc);
       compute_nu(g, nu, cs2, M_star, alpha);
       calc_gas_velocities(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor, Rcav);
   }

Sound speed
-----------

The sound speed squared for an ideal gas with mean molecular weight *μ* is:

.. code-block:: cpp

   void compute_cs2(const Grid &g, Field<double> &T, Field<double> &cs2, double mu) {
       for (int i = 0; i < g.NR + 2*g.Nghost; i++)
           for (int j = 0; j < g.Nphi + 2*g.Nghost; j++)
               cs2(i,j) = R_gas * T(i,j) / mu;
   }

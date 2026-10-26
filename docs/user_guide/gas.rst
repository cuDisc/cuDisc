.. _user_guide_gas:

Gas Structure and Evolution
===========================

cuDisc treats the gas as a background that evolves on a 1D radial grid,
with the 2D vertical structure reconstructed from hydrostatic equilibrium.
This section describes how to set up the gas, compute the hydrostatic
equilibrium, and evolve the gas surface density over time.

The Star object
---------------

Stellar properties are encapsulated in a :cpp:class:`Star` object:

.. code-block:: cpp

   double M_star  = 1.;              // stellar mass in solar masses
   double T_star  = 4500.;           // effective temperature (K)
   double R_star  = 1.7 * Rsun;      // stellar radius (cm)
   double L_star  = 4. * M_PI * sigma_SB * std::pow(T_star, 4.) * std::pow(R_star, 2.);

   Star star(GMsun * M_star, L_star, T_star);

   // Set stellar spectrum for radiative transfer
   int num_wavelengths = 100;
   star.set_wavelengths(num_wavelengths, opacs.lam());
   star.set_blackbody_fluxes();   // use a Planck spectrum

Gas surface density
--------------------

The 1D gas surface density is stored in a :cpp:class:`CudaArray`:

.. code-block:: cpp

   CudaArray<double> Sig_g = make_CudaArray<double>(g.NR + 2*g.Nghost);

Initialise it with your chosen radial profile. For a power-law disc:

.. code-block:: cpp

   double Sig0 = 100.;   // surface density at 1 au, g cm^-2
   double p    = -1.;    // power-law index

   for (int i = 0; i < g.NR + 2*g.Nghost; i++) {
       Sig_g[i] = Sig0 * std::pow(g.Rc(i) / au, p);
   }

Hydrostatic equilibrium
------------------------

The 2D density structure of the gas is computed from hydrostatic equilibrium
given the surface density and temperature field:

.. code-block:: cpp

   compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g);

This fills the ``rho`` and velocity components of ``Ws_g`` at every cell.
The sound speed squared ``cs2`` must be up to date before calling this
function. When dust is included, its weight can optionally be accounted for:

.. code-block:: cpp

   compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, Ws_d, gas_floor, dust_floor);

Gas velocities
--------------

The radial and azimuthal gas velocities follow from the pressure gradient and
viscous torque. After calling ``compute_hydrostatic_equilibrium``, compute
the gas velocities:

.. code-block:: cpp

   CudaArray<double> nu = make_CudaArray<double>(g.NR + 2*g.Nghost);

   // Kinematic viscosity from alpha prescription
   for (int i = 0; i < g.NR + 2*g.Nghost; i++) {
       double Om = std::sqrt(GMsun * M_star / std::pow(g.Rc(i), 3.));
       nu[i] = alpha * cs2(i, g.Nghost) / Om;
   }

   int gas_boundary = BoundaryFlags::open_R_inner
                    | BoundaryFlags::open_R_outer
                    | BoundaryFlags::open_Z_outer;
   double gas_floor = 1.e-30;    // minimum gas density (g cm^-3)

   calc_gas_velocities(g, Sig_g, Ws_g, cs2, nu, alpha, star,
                       gas_boundary, gas_floor, Rcav);

where ``Rcav`` is the cavity radius (set to zero if there is no cavity).

1D gas evolution
-----------------

The gas surface density can be evolved in time using the 1D viscous disc
equation. The ``gas1d.h`` header provides this functionality. A typical
update within the time loop looks like:

.. code-block:: cpp

   // Update gas surface density
   update_gas_sigma(g, Sig_g, nu, dt, gas_boundary);

   // Photoevaporative wind loss
   CudaArray<double> Sigdot_wind = make_CudaArray<double>(g.NR + 2*g.Nghost);
   Sigdot_wind_PicPD(g, Sigdot_wind, logLx);   // Picogna et al. 2019 profile

   // Recompute hydrostatic structure
   compute_cs2(g, T, cs2, mu);
   compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g);
   calc_gas_velocities(g, Sig_g, Ws_g, cs2, nu, alpha, star,
                       gas_boundary, gas_floor, Rcav);

Boundary conditions
--------------------

Boundary conditions for gas quantities are specified using bitwise flags
from the ``BoundaryFlags`` enum:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Flag
     - Effect
   * - ``BoundaryFlags::open_R_inner``
     - Outflow at the inner radial boundary.
   * - ``BoundaryFlags::open_R_outer``
     - Outflow at the outer radial boundary.
   * - ``BoundaryFlags::open_Z_outer``
     - Outflow at the upper polar boundary.
   * - ``BoundaryFlags::closed_R_inner``
     - Reflecting (zero-flux) inner radial boundary.
   * - ``BoundaryFlags::closed_Z_outer``
     - Reflecting upper polar boundary.

Flags are combined with the bitwise-or operator:

.. code-block:: cpp

   int boundary = BoundaryFlags::open_R_inner
                | BoundaryFlags::open_R_outer
                | BoundaryFlags::open_Z_outer;

.. _user_guide_dust:

Dust Dynamics
=============

cuDisc represents the dust as a collection of *N*:sub:`spec` pressureless
fluid species, one per grain-size bin. Each species has its own density and
velocity field. The :cpp:class:`DustDynamics` solver advances all species
simultaneously using a second-order Godunov scheme.

The size grid
-------------

Grain sizes are defined on a logarithmic grid spanning from a minimum grain
radius *a*:sub:`0` to a maximum *a*:sub:`1`, with *n*:sub:`spec` bins:

.. code-block:: cpp

   int    n_spec  = 135;
   double rho_p   = 1.6;      // grain material density (g cm^-3)
   double a0      = 1.e-5;    // minimum grain size (cm)
   double a1      = 20.;      // maximum grain size (cm)

   SizeGrid sizes(g, a0, a1, n_spec, rho_p);

The ``SizeGrid`` object stores the grain radius, mass, and material density
for each bin, and provides them to the coagulation kernel and opacity
interpolation routines.

Setting initial conditions
--------------------------

Allocate the dust primitive-variable field and diffusion-coefficient field:

.. code-block:: cpp

   Field3D<Prims>  Ws_d = create_field3D<Prims>(g, n_spec);
   Field3D<double> D    = create_field3D<double>(g, n_spec);

A typical initialisation distributes dust with an MRN-like surface density
profile and a Gaussian vertical distribution set by the dust scale height:

.. code-block:: cpp

   double d_to_g = 0.01;    // dust-to-gas mass ratio
   double Sc     = 1.;      // Schmidt number

   for (int i = 0; i < g.NR + 2*g.Nghost; i++) {
       double Om  = std::sqrt(GMsun * M_star / std::pow(g.Rc(i), 3.));
       double h_g = cs(i, g.Nghost) / Om;   // gas scale height

       for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
           for (int k = 0; k < n_spec; k++) {
               double St   = sizes.grain_props(i,j,k).a * M_PI / 2.
                             * sizes.solid_density() / Sig_g[i];
               double h_d  = h_g * std::sqrt(alpha / (alpha + St));
               double Sig_d = /* your initial surface density per bin */ ;
               Ws_d(i,j,k).rho   = Sig_d / (std::sqrt(2.*M_PI) * h_d)
                                    * std::exp(-g.Zc(i,j)*g.Zc(i,j) / (2.*h_d*h_d));
               Ws_d(i,j,k).v_R   = 0.;
               Ws_d(i,j,k).v_phi = std::sqrt(GMsun * M_star / g.Rc(i));
               Ws_d(i,j,k).v_Z   = 0.;
               D(i,j,k) = Ws_g(i,j).rho * alpha * cs2(i,j) / (Sc * Om);
           }
       }
   }

Constructing the solver
------------------------

The dust dynamics solver is created by passing the diffusion field, sound
speed, a source-term object, and the CFL coefficients:

.. code-block:: cpp

   double C_adv  = 0.4;     // advective CFL coefficient
   double C_diff = 0.2;     // diffusive CFL coefficient
   double floor  = 1.e-10;  // dust density floor (fraction of gas density)
   double gas_floor = 1.e-30;

   Sources src(T, Ws_g, sizes, floor, M_star, mu2D);
   DustDynamics dyn(D, cs, src, C_adv, C_diff, floor, gas_floor);

Setting boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   int boundary = BoundaryFlags::open_R_inner
                | BoundaryFlags::open_R_outer
                | BoundaryFlags::open_Z_outer;
   dyn.set_boundaries(boundary);

The time step
~~~~~~~~~~~~~

Get the CFL-limited time step before the main loop:

.. code-block:: cpp

   double dt_CFL = dyn.get_CFL_limit(g, Ws_d, Ws_g);

Advancing the dust
-------------------

Call the dust dynamics solver once per time step:

.. code-block:: cpp

   dyn(g, Ws_d, Ws_g, dt);

After advancing, apply the gas-floor mask to prevent dust accumulating in
cells where the gas density has fallen below the floor:

.. code-block:: cpp

   CudaArray<double> h_floor = make_CudaArray<double>(g.NR + 2*g.Nghost);
   dyn.compute_gas_floor_height(g, Ws_g, h_floor);
   dyn.floor_above(g, Ws_d, Ws_g, h_floor);

Updating the diffusion coefficient
------------------------------------

After any update to the gas density or temperature, recompute the
diffusion coefficients before the next dynamics step:

.. code-block:: cpp

   for (int i = 0; i < g.NR + 2*g.Nghost; i++) {
       double Om = std::sqrt(GMsun * M_star / std::pow(g.Rc(i), 3.));
       for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
           for (int k = 0; k < n_spec; k++) {
               D(i,j,k) = Ws_g(i,j).rho * alpha * cs2(i,j) / (Sc * Om);
           }
       }
   }

Restart capability
------------------

The dynamics solver writes a restart file that records the current active
region. On restart, read it back immediately after setting up the solver:

.. code-block:: cpp

   // On fresh start — nothing to do.

   // On restart:
   dyn.read_active(g, dir);   // restore the active-cell mask from disk

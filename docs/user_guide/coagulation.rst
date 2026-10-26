.. _user_guide_coagulation:

Dust Coagulation and Fragmentation
====================================

cuDisc solves the Smoluchowski coagulation equation to evolve the full
grain-size distribution. Coagulation and fragmentation are treated with a
piecewise-constant kernel derived from the relative velocities between grains.

Setting up the kernel
----------------------

The default kernel is the Birnstiel coagulation kernel, which computes
collision rates from the turbulent, Brownian, radial-drift, and settling
velocity components:

.. code-block:: cpp

   Field<double> alpha2D = create_field<double>(g);
   Field<double> mu2D    = create_field<double>(g);

   for (int i = 0; i < g.NR + 2*g.Nghost; i++)
       for (int j = 0; j < g.Nphi + 2*g.Nghost; j++) {
           alpha2D(i,j) = alpha;
           mu2D(i,j)    = mu;        // mean molecular weight
       }

   BirnstielKernel kernel(g, sizes, Ws_d, Ws_g, cs, alpha2D, mu2D);

Setting the fragmentation velocity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fragmentation velocity *v*:sub:`frag` (cm s\ :sup:`-1`) is the maximum
relative speed at which grains stick. Collisions above this speed result in
fragmentation:

.. code-block:: cpp

   kernel.set_fragmentation_threshold(1000.);   // 10 m/s in cm/s

Setting up the integrator
--------------------------

Coagulation is advanced with a Bogacki–Shampine 3(2) adaptive Runge–Kutta
integrator:

.. code-block:: cpp

   double rtol = 1.e-2;    // relative tolerance
   double atol = 1.e-10;   // absolute tolerance

   BS32Integration<CoagulationRate<decltype(kernel), SimpleErosion>>
       coagulation_integrate(
           create_coagulation_rate(
               sizes,
               kernel,
               SimpleErosion(1, 11./6., sizes.min_mass())  // erosion model
           ),
           rtol, atol
       );

The ``SimpleErosion`` model distributes the mass from a fragmentation event
as a power-law distribution of smaller fragments. Its arguments are:

- ``n_frag``: number of fragments produced per collision.
- ``power``: the power-law index of the fragment mass distribution.
- ``min_mass``: the minimum fragment mass (set to the mass of the smallest
  size bin).

Advancing coagulation
----------------------

Coagulation is advanced over a time interval ``dt_coag``:

.. code-block:: cpp

   coagulation_integrate.integrate(g, Ws_d, Ws_g, dt_coag, dt_coag_last, floor);

.. note::

   Coagulation does not need to be called every dynamics time step — it is
   typically much slower than the dynamical timescale and can be called every
   few to hundreds of dynamics steps. The ``steadyTD`` example uses an
   adaptive scheme that calls coagulation when the dynamical time step has
   advanced by at least ``dt_coag_last``.

Updating the kernel
--------------------

The coagulation kernel depends on the gas properties and the sound speed,
which change when the temperature is updated. After each temperature update,
the kernel is refreshed automatically when ``integrate`` is called, because
it holds references to the live ``Ws_g`` and ``cs`` fields.

Ice in coagulation (ice–vapour chemistry)
------------------------------------------

When ice–vapour chemistry is enabled (see :doc:`../physics/ice_chemistry`),
the coagulation module is augmented to track the ice mass fraction in each
grain-size bin. Ice mass is transferred between bins in proportion to the
ice mass fraction of the colliding grains. The opacity is then computed from
the mixed ice–dust composition using the Maxwell-Garnett effective medium
theory.

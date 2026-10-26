.. _api_coagulation:

Coagulation
===========

Defined in ``headers/coagulation.h``.

BirnstielKernel
---------------

Evaluates the coagulation kernel *K*(*m*\ :sub:`1`, *m*\ :sub:`2`) following
the prescription of Birnstiel et al. (2010). Relative velocities include
Brownian motion, turbulence, radial drift, and differential settling.

.. code-block:: cpp

   class BirnstielKernel;

BirnstielKernel constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   BirnstielKernel(const Grid&           g,
                   const SizeGrid&       sizes,
                   const Field3D<Prims>& Ws_d,
                   const Field<Prims>&   Ws_g,
                   const Field<double>&  cs,
                   const Field<double>&  alpha,
                   const Field<double>&  mu);

Methods
~~~~~~~

.. code-block:: cpp

   // Set the fragmentation velocity [cm/s]
   void set_fragmentation_threshold(double v_frag);

SimpleErosion
-------------

Fragmentation model that distributes fragments as a power-law in mass.

.. code-block:: cpp

   class SimpleErosion;

SimpleErosion constructor
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   SimpleErosion(double p_erosion,
                 double xi,
                 double m_min);

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``p_erosion``
     - Fraction of colliding mass distributed to fragments (default 1).
   * - ``xi``
     - Fragment mass power-law index (default 11/6).
   * - ``m_min``
     - Minimum fragment mass [g] (typically ``sizes.min_mass()``).

CoagulationRate
---------------

Combines a kernel and a fragmentation model into a single callable that
computes the right-hand side of the discrete Smoluchowski equation.

.. code-block:: cpp

   template <typename Kernel, typename Fragmentation>
   class CoagulationRate;

Factory function
~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <typename Kernel, typename Fragmentation>
   CoagulationRate<Kernel, Fragmentation>
   create_coagulation_rate(const SizeGrid& sizes,
                           Kernel& kernel,
                           Fragmentation frag);

BS32Integration
---------------

Bogacki–Shampine 3(2) adaptive Runge–Kutta integrator for the coagulation
ODE.

.. code-block:: cpp

   template <typename RHS>
   class BS32Integration;

BS32Integration constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   BS32Integration(RHS rhs, double rtol, double atol);

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``rhs``
     - The right-hand side functor (e.g. a ``CoagulationRate`` instance).
   * - ``rtol``
     - Relative tolerance (default 1e-2).
   * - ``atol``
     - Absolute tolerance (default 1e-10).

Advance over a time interval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   void integrate(const Grid&           g,
                  Field3D<Prims>&       Ws_d,
                  const Field<Prims>&   Ws_g,
                  double                dt_coag,
                  const SizeGrid&       sizes);

Integrates the coagulation equation at each grid cell independently over
the interval ``[0, dt_coag]``.

Typical usage
~~~~~~~~~~~~~

.. code-block:: cpp

   #include "coagulation.h"
   #include "dustdyn.h"

   // Set up kernel and integrator (once, before time loop)
   BirnstielKernel kernel(g, sizes, Ws_d, Ws_g, cs, alpha, mu);
   kernel.set_fragmentation_threshold(1000.);   // 10 m/s

   auto rate = create_coagulation_rate(sizes, kernel, SimpleErosion(1, 11./6., sizes.min_mass()));
   BS32Integration<decltype(rate)> coag(rate, 1e-2, 1e-10);

   // Inside the time loop
   if (t_since_coag >= dt_coag) {
       coag.integrate(g, Ws_d, Ws_g, dt_coag, sizes);
       t_since_coag = 0.;
   }

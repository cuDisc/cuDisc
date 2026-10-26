.. _user_guide_timestepping:

Time-stepping
=============

cuDisc uses operator splitting to advance the simulation in time. Each
physical process is updated in sequence within a single time step. The order
used in the example simulations is:

1. Dust advection–diffusion (:cpp:class:`DustDynamics`)
2. Dust coagulation/fragmentation (coagulation integrator, called at a lower
   cadence)
3. Radiative transfer / temperature update (:cpp:class:`FLD_Solver`, called
   at a lower cadence)

The CFL time step
------------------

The time step is limited by both advective and diffusive CFL conditions:

.. math::

   \Delta t = \min\!\bigl(C_\text{adv}\,\Delta t_\text{adv},\;
                           C_\text{diff}\,\Delta t_\text{diff}\bigr)

with default coefficients *C*:sub:`adv` = 0.4 and *C*:sub:`diff` = 0.2. The
CFL time step is computed before the main loop:

.. code-block:: cpp

   double dt_CFL = dyn.get_CFL_limit(g, Ws_d, Ws_g);

During the loop, the time step is allowed to increase by at most 10% per
step, capped by both the CFL limit and the time to the next output snapshot:

.. code-block:: cpp

   dt = std::min(dt_CFL, 1.1 * dt);    // grow by at most 10%
   dt = std::min(dt, t_next - t);       // do not overshoot next output

The main time loop
------------------

A minimal time loop looks like:

.. code-block:: cpp

   double t = 0., dt = dt_CFL;
   double t_coag = 0., dt_coag = 0.;
   double t_temp = 0.;

   for (double t_out : output_times) {

       while (t < t_out) {

           // --- Dust dynamics ---
           dt = std::min(dt_CFL, 1.1 * dt);
           dt = std::min(dt, t_out - t);

           dyn(g, Ws_d, Ws_g, dt);
           dyn.floor_above(g, Ws_d, Ws_g, h_floor);

           // --- Coagulation (lower cadence) ---
           if (need_coag_step(t, dt, t_coag, dt_coag)) {
               cs2_to_cs(g, cs, cs2);
               coagulation_integrate.integrate(g, Ws_d, Ws_g,
                                               (t + dt) - t_coag,
                                               dt_coag, floor);
               t_coag = t + dt;
           }

           // --- Temperature (lower cadence) ---
           if (need_temp_step(t, dt, t_temp)) {
               // recompute opacities, stellar heating, call FLD_Solver ...
               t_temp = t + dt;
               // recompute cs2, hydrostatic equilibrium, gas velocities ...
           }

           // Update dt_CFL after any change to the velocity field
           dt_CFL = dyn.get_CFL_limit(g, Ws_d, Ws_g);
           t += dt;
           count++;
       }

       write_file(dir, Nout++, g, Ws_d, Ws_g, Sig_g, T, J);
   }

Coagulation cadence
~~~~~~~~~~~~~~~~~~~~

In the example simulations, coagulation is called when any of the following
is true:

- The dynamics time ``t + dt`` has advanced past ``t_coag + dt_coag`` (the
  next scheduled coagulation step).
- The simulation is approaching the next output snapshot (``dt == t_out - t``).
- More than 50 years have elapsed since the last coagulation step.

This heuristic keeps the coagulation step reasonably synchronised with the
dynamics without calling it every step.

Temperature cadence
~~~~~~~~~~~~~~~~~~~~

The temperature is updated when the time since the last temperature update
exceeds 10% of the time over which the temperature changed by 1%
(``dt_1perc``), or when the absolute elapsed time exceeds a maximum interval
(``dt_temp_max``, typically 5000 yr). If the FLD solver fails to converge
(fractional error above 10% after two iterations), the temperature update is
skipped for that step and ``dt_1perc`` is increased.

Performance notes
-----------------

On an NVIDIA A100 GPU, the example 2D disc simulation achieves approximately
100–500 years per second wall-clock time, depending on grid resolution and
the frequency of temperature updates. GPU utilisation is highest during the
dust dynamics steps. The FLD solver is the most computationally demanding
single operation.

The ``years per second`` metric is printed every 1000 time steps:

.. code-block:: text

   t = 1.0e5 years
   Years per second: 312.7

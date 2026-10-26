.. _physics_coagulation:

Dust Coagulation and Fragmentation
====================================

cuDisc solves the Smoluchowski coagulation equation to evolve the grain-size
distribution self-consistently with the dust dynamics. The code follows the
approach of Birnstiel et al. (2010) and Stammler & Birnstiel (2022) for the
coagulation kernel, but couples it to the full 2D spatial structure.

The Smoluchowski equation
--------------------------

The evolution of the number density :math:`n(m)` of grains of mass *m* is
governed by the Smoluchowski coagulation equation:

.. math::

   \frac{\partial n(m)}{\partial t}
   &= \frac{1}{2}\int_0^m K(m', m-m')\,n(m')\,n(m-m')\,\mathrm{d}m' \\
   &\quad - n(m)\int_0^\infty K(m, m')\,n(m')\,\mathrm{d}m' \\
   &\quad + \text{fragmentation terms}.

The first two terms describe the gain and loss of grains of mass *m* through
coagulation. The fragmentation terms add grains of mass *m* produced by
fragmenting larger grains and remove grains of mass *m* that fragment when
hit by other grains.

Coagulation kernel
-------------------

The coagulation kernel *K*(*m*\ :sub:`1`, *m*\ :sub:`2`) is the rate at which
two grains of masses *m*\ :sub:`1` and *m*\ :sub:`2` collide. It is the
product of the collision cross-section and the relative velocity:

.. math::

   K(m_1, m_2) = p_\text{coag}\,\pi(s_1 + s_2)^2\,\Delta v(m_1, m_2),

where *s*\ :sub:`1,2` are the grain radii and *p*\ :sub:`coag` is the
sticking probability (set to 1 at low velocities). The relative velocity
:math:`\Delta v` is computed from the quadrature sum of contributions from:

- **Brownian motion**: dominant for sub-micron grains.
- **Turbulence**: dominant for mm-cm grains in typical disc conditions.
- **Differential radial drift**: relevant in the drift-dominated regime.
- **Differential settling**: relevant for large grains near the midplane.

Fragmentation
--------------

When the relative velocity between two colliding grains exceeds the
**fragmentation velocity** *v*\ :sub:`frag`, the collision results in
fragmentation rather than sticking. cuDisc uses the ``SimpleErosion`` model,
in which the mass of the two colliding grains is redistributed as a power-law
distribution of smaller fragments:

.. math::

   \frac{\mathrm{d}N}{\mathrm{d}m} \propto m^{-\xi}, \quad
   m_\text{min} \le m \le m_1 + m_2,

with default power-law index *ξ* = 11/6 and minimum fragment mass equal to
the mass of the smallest grid bin. This distribution is chosen to be
consistent with the observed power-law grain-size distributions in
debris-disc and asteroid-belt collisional cascades.

The fragmentation probability is a step function in relative velocity:

.. math::

   p_\text{coag}(\Delta v) =
   \begin{cases}
     1 & \Delta v < v_\text{frag}, \\
     0 & \Delta v \ge v_\text{frag}.
   \end{cases}

Numerical method
-----------------

The discrete coagulation equation is solved on a logarithmic mass grid with
:math:`n_\text{spec}` bins. The coagulation kernel is evaluated pairwise for
all (:math:`k_1`, :math:`k_2`) combinations at the start of a coagulation step
and reused within the step.

Time integration uses the **Bogacki–Shampine 3(2) Runge–Kutta** scheme, an
adaptive-step ODE integrator with embedded error estimation:

- The relative and absolute tolerances control the step size.
- The integrator accepts the time interval passed to it and internally
  sub-steps as needed to meet the tolerances.
- The method is third-order accurate with second-order error estimation.

This scheme is efficient for stiff coagulation problems because it can take
large steps when the grain-size distribution changes slowly, and small steps
only when needed.

Operator splitting with dynamics
----------------------------------

Coagulation and dynamics are split with Strang splitting. Dust dynamics
advances the density field over a time step Δ*t*, then coagulation is called
to evolve the size distribution locally (at each grid cell independently)
over the same Δ*t*. Because the coagulation timescale is generally longer
than the dynamical CFL time step, coagulation is called at a lower cadence —
typically every few to hundreds of dynamics steps.

Comparison with DustPy
-----------------------

The cuDisc coagulation module was verified against the DustPy code
(Stammler & Birnstiel 2022) for the case of a 1D vertically integrated disc
without radial drift. The two codes agree on the steady-state grain-size
distribution to within a few percent across the full size range for both the
fragmentation-limited and drift-limited regimes. The comparison test is
included as ``test_coagdustpy`` (see :ref:`tests/index:Tests`).

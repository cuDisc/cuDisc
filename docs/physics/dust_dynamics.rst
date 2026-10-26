.. _physics_dust_dynamics:

Dust Dynamics
=============

cuDisc treats each grain-size bin as an independent pressureless fluid that
is advected by its own velocity field and diffused by turbulence. The gas
provides a background through which the dust moves under aerodynamic drag.

Governing equations
--------------------

The conserved variables for a single dust species are

.. math::

   \mathbf{Q} = \bigl(\rho,\;\rho v_R,\;\rho v_\phi R,\;\rho v_Z\bigr)^\top,

where *ρ* is the dust mass density and (v\ :sub:`R`, v\ :sub:`φ`, v\ :sub:`Z`)
are the velocity components in cylindrical coordinates. The angular momentum
:math:`\rho v_\phi R` is the conserved azimuthal quantity (Robinson et al.
2024, Eq. 4).

The evolution equation is

.. math::

   \frac{\partial \mathbf{Q}}{\partial t}
   + \frac{1}{V}\sum_\text{faces} \mathbf{F}\,A = \mathbf{S},

where **F** is the numerical flux through each face, *A* is the face area,
and **S** collects the source terms (drag, gravity, diffusion).

Operator splitting
-------------------

The update is split into a **transport step** (hyperbolic fluxes only) and a
**source step** (drag, gravity, diffusion). This is a standard Godunov
operator-splitting approach, which is second-order in time when the two
substeps are applied symmetrically (Strang splitting). The time update
is (Eq. 8):

.. math::

   \mathbf{Q}^{n+1} = \mathbf{Q}^n
   - \frac{\Delta t}{V}\sum_\text{faces}\mathbf{F}^{n+1/2}\,A

where the half-time fluxes **F**\ :sup:`n+1/2` are computed using a
predictor–corrector approach.

Transport solver
-----------------

The transport step uses a **second-order piecewise-linear reconstruction**
with a modified van Leer slope limiter (Mignone 2014). The algorithm is:

1. **Predictor**: First-order donor-cell (upwind) reconstruction and Riemann
   fluxes.
2. **Half-step update**: Advance the cell averages to time *n + 1/2* using
   the first-order fluxes.
3. **Corrector**: Reconstruct the cell averages with piecewise-linear slopes,
   limiting with the modified van Leer limiter.
4. **Final fluxes**: Compute the Riemann fluxes from the second-order
   reconstructed states at time *n + 1/2*.

Delta-shock treatment
~~~~~~~~~~~~~~~~~~~~~

Pressureless dust fluids are susceptible to delta-shock formation when two
streams collide. cuDisc handles this by using the Roe-averaged velocity
(Eq. 11):

.. math::

   v_\text{Roe} = \frac{\sqrt{\rho_L}\,v_L + \sqrt{\rho_R}\,v_R}
                       {\sqrt{\rho_L} + \sqrt{\rho_R}}

and an upwind flux at faces where the Roe velocity is not well-defined
(LeVeque 2004; Eq. 12).

Aerodynamic drag
-----------------

The drag force per unit volume on a dust species is (Eq. 13):

.. math::

   \mathbf{f}_\text{drag} = -\frac{\rho}{t_s}\bigl(\mathbf{v} - \mathbf{v}_g\bigr),

where **v** and **v**\ :sub:`g` are the dust and gas velocities, and
*t*\ :sub:`s` is the stopping time. In the Epstein drag regime (grain radius
*s* much smaller than the mean free path), the stopping time is (Eq. 14):

.. math::

   t_s = \frac{\rho_m\,s}{\rho_g\,v_\text{th}},
   \quad
   v_\text{th} = \sqrt{\frac{8}{\pi}}\,c_s,

where *ρ*:sub:`m` is the grain material density and *c*:sub:`s` is the
isothermal sound speed.

.. note::

   cuDisc currently implements the Epstein regime only. Stokes drag (relevant
   for particles with *s* larger than the mean free path) is not yet included.

Drag is **integrated implicitly** so that arbitrarily large stopping times
(St ≫ 1) are stable. The Jacobian of the drag term is linear and diagonal,
so the implicit update has an exact analytic form.

Turbulent diffusion
--------------------

Turbulent diffusion transports dust relative to the gas. The diffusion flux
for species *i* is (Eq. 15):

.. math::

   \mathbf{F}_\text{diff} = -\frac{\rho_g\,\nu}{\mathrm{Sc}_i}
   \,\nabla\!\left(\frac{\rho_i}{\rho_g}\right),

where *ν* is the kinematic viscosity and Sc\ :sub:`i` is the Schmidt number
(set to 1 in the standard runs). The diffusion flux is discretised using the
second-order scheme of Wu, Gao & Dai (2012).

Diffusion is also integrated implicitly to allow time steps larger than the
diffusive CFL limit.

CFL condition
--------------

The time step is limited by both the advective and diffusive CFL conditions
(Eqs. 17–19):

.. math::

   \Delta t_\text{adv}  &= C_\text{adv}\,\min_{i,j}
   \left(\frac{\Delta R_{ij}}{|v_{R,ij}|},\;
         \frac{\Delta Z_{ij}}{|v_{Z,ij}|}\right), \\[4pt]
   \Delta t_\text{diff} &= C_\text{diff}\,\min_{i,j}
   \left(\frac{(\Delta R_{ij})^2}{D_{R,ij}},\;
         \frac{(\Delta Z_{ij})^2}{D_{Z,ij}}\right),

with default coefficients *C*:sub:`adv` = 0.4 and *C*:sub:`diff` = 0.2.

Azimuthal (2.5D) treatment
----------------------------

Although cuDisc is a 2D code (it does not resolve the azimuthal structure),
the azimuthal component of the momentum equation is solved in an
angular-momentum-conserving form. This is sometimes called "2.5D" and
accounts for the azimuthal pressure gradient (sub-Keplerian rotation of the
gas) and the geometric correction to the Coriolis term.

Gas evolution
--------------

The gas is evolved in the radial direction using the standard 1D viscous disc
equation. The vertically integrated form is

.. math::

   \frac{\partial \Sigma_g}{\partial t}
   = \frac{1}{R}\frac{\partial}{\partial R}\!\left[
     3\sqrt{R}\,\frac{\partial}{\partial R}\!\left(\nu\Sigma_g\sqrt{R}\right)
   \right] + \dot\Sigma_\text{wind},

where *ν* = *α c*\ :sub:`s`\ *H*:sub:`g` is the Shakura–Sunyaev viscosity
and *Σ̇*\ :sub:`wind` is a mass-loss rate from a photoevaporative wind. The
vertical structure is reconstructed from hydrostatic equilibrium.

Hydrostatic equilibrium
~~~~~~~~~~~~~~~~~~~~~~~~~

The gas density at each cell is set by the condition

.. math::

   \frac{\partial P}{\partial Z}
   = -\frac{G M_\star \rho_g\,Z}{\bigl(R^2 + Z^2\bigr)^{3/2}},

which is solved for the density profile given the surface density and the
sound speed.

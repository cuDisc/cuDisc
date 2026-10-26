.. _physics_radiative_transfer:

Radiative Transfer
==================

Disc temperatures in cuDisc are computed self-consistently by solving the
frequency-dependent radiative transfer problem in 2D. The method combines
ray-tracing for the direct stellar irradiation with a flux-limited diffusion
(FLD) approximation for thermal re-emission and scattering.

The hybrid approach
--------------------

Computing the full frequency-resolved radiative transfer problem in 2D on
every time step is computationally prohibitive. cuDisc therefore splits the
radiation field into two components:

1. **Direct stellar irradiation** is computed with frequency-resolved
   ray-tracing. This gives an exact treatment of the direct stellar heating
   and single-scattered light.
2. **Thermal re-emission** (dust thermal emission and the diffuse radiation
   field) is computed with a multifrequency FLD solver. FLD is a
   computationally efficient approximation to the full radiative transfer
   equation that works well in both optically thick (diffusion) and optically
   thin (free-streaming) limits.

Energy equations
-----------------

The internal energy of the gas-dust mixture and the radiation field are
coupled through the equations (Robinson et al. 2024, Eqs. 39–40):

.. math::

   \frac{\partial (\rho C_v T)}{\partial t}
   &= -\nabla\cdot\mathbf{F}_\text{diff}
      + \sum_\nu \rho\kappa_\nu^P \bigl(4\pi B_\nu - c E_\nu\bigr)
      + \Gamma_\text{visc}, \\[4pt]
   \frac{\partial E_\nu}{\partial t}
   &= \nabla\cdot\!\left(\frac{c}{3\rho\kappa_\nu}\nabla E_\nu\right)
      - \rho\kappa_\nu^P \bigl(c E_\nu - 4\pi B_\nu\bigr)
      + S_\nu^\text{star},

where *E*\ :sub:`ν` = *cJ*\ :sub:`ν` is the frequency-specific radiation
energy density (proportional to the mean intensity *J*\ :sub:`ν`),
*B*\ :sub:`ν` is the Planck function, *κ*\ :sub:`ν`\ :sup:`P` is the Planck-mean
opacity, and Γ:sub:`visc` is the viscous heating rate.

Flux-limited diffusion
-----------------------

The FLD approximation replaces the exact radiative flux with

.. math::

   \mathbf{F}_\nu = -\frac{c\,\lambda_\nu}{\rho\kappa_\nu}\,\nabla E_\nu,

where *λ*\ :sub:`ν` is the flux limiter. cuDisc uses the Kley (1989) limiter
(Eqs. 42–43), which interpolates between the diffusion regime
(*λ* = 1/3) and the free-streaming regime (*λ → c* \|∇E\| / E):

.. math::

   \lambda_\nu = \frac{1}{R_\nu}\!\left(\coth R_\nu - \frac{1}{R_\nu}\right),
   \quad
   R_\nu = \frac{|\nabla E_\nu|}{\rho\kappa_\nu E_\nu}.

Multifrequency treatment
~~~~~~~~~~~~~~~~~~~~~~~~

The FLD equation is solved on ``n_bands`` frequency bands rather than a
single grey opacity. Each band has its own opacity and radiation energy
density, and the bands are coupled through the Planck source term
:math:`\rho\kappa_\nu^P(cE_\nu - 4\pi B_\nu)`. The Planck-weighted opacity
in each band is recomputed at each temperature update to account for the
temperature-dependent dust emission spectrum.

Stellar irradiation
--------------------

The stellar heating rate *S*\ :sup:`star` is computed by ray-tracing radially
outward from the star (Eq. 45). For each frequency bin in the stellar
spectrum, a ray is cast from the star, and the optical depth along the ray is
accumulated. The absorbed power per unit volume at each cell is

.. math::

   S_\nu^\text{star} = \rho\kappa_\nu^P\,F_\nu^\star\,
   e^{-\tau_\nu},

where *F*\ :sub:`ν`\ :sup:`★` is the stellar flux and *τ*\ :sub:`ν` is the
optical depth from the star to the cell.

Scattered light is handled by computing the first-order scattering source:

.. math::

   S_\nu^\text{sca} = \rho\kappa_\nu^\text{sca}\,F_\nu^\star\,
   e^{-\tau_\nu^\text{ext}},

where *κ*\ :sub:`ν`\ :sup:`sca` is the scattering opacity and
*τ*\ :sub:`ν`\ :sup:`ext` is the optical depth to extinction. This
single-scattering source is then passed to the FLD solver as an additional
heating term.

Viscous heating
---------------

Turbulent viscous dissipation heats the gas at a rate

.. math::

   \Gamma_\text{visc}
   = \frac{9}{4}\,\nu\,\Omega_K^2\,\rho_g,

where *Ω*\ :sub:`K` is the Keplerian angular velocity. This is added to the
right-hand side of the internal energy equation at each cell.

Opacities
---------

cuDisc uses frequency-resolved dust opacities from the DSHARP opacity table
(Birnstiel et al. 2018), which provides absorption and scattering cross-sections
as a function of grain size and wavelength. The total opacity field is computed
from the sum over all grain-size bins and the gas contribution:

.. math::

   \rho\kappa_\nu = \sum_k \rho_{d,k}\,\kappa_{\nu,k}^\text{abs}
                    + \rho_g\,\kappa_{\nu,g},

where the gas opacity *κ*\ :sub:`ν,g` is set to a fixed small floor
(``set_k_g_min_grain``).

Temperature convergence
------------------------

At the start of a simulation, and whenever the dust distribution changes
significantly, the temperature must be iterated to self-consistency. A
typical iteration updates the opacities, re-solves the FLD equation, and
then re-solves the hydrostatic equilibrium until the fractional change in
temperature falls below 10\ :sup:`-5`.

Benchmarks
-----------

The temperature solver has been validated against the Pinte et al. (2009)
benchmark, which provides a reference temperature structure computed with a
full Monte Carlo radiative transfer code for a flared disc with a single grain
size. The cuDisc FLD result agrees with the reference to within a few percent
over most of the disc. Details of this benchmark are given in
:ref:`tests/index:Tests`.

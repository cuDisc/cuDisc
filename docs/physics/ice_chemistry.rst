.. _physics_ice_chemistry:

Ice–Vapour Chemistry
======================

cuDisc includes a module for the adsorption and desorption of volatile
molecular ices (e.g. CO, H\ :sub:`2`\ O) onto and off dust grains. The
ice–vapour chemistry is fully coupled to the 2D dust dynamics and
grain-size distribution, enabling studies of snow-line structure and
volatile transport in protoplanetary discs.

The methods are described in Robinson, Owen & Booth (2026).

Physical model
---------------

The total mass of a volatile species is split between the vapour phase
(density :math:`\rho_v`) and the ice phase (density :math:`\rho_i` per
grain-size bin). The two phases are coupled by three processes:

1. **Adsorption**: vapour molecules collide with grains and freeze onto
   the surface.
2. **Thermal desorption**: ice sublimates when the surface temperature is
   high enough.
3. **Photodesorption**: UV photons (from the stellar radiation field or
   cosmic rays) dislodge ice molecules from grain surfaces.

Adsorption rate
~~~~~~~~~~~~~~~~

The adsorption rate onto grains of number density *n* with radius *a* is
(Robinson et al. 2026, Eq. 3):

.. math::

   R_\text{ad} = \pi a^2\,v_\text{th}\,n\,\rho_v,
   \quad
   v_\text{th} = \sqrt{\frac{8 k_B T}{\pi m_\text{mol}}},

where *m*\ :sub:`mol` is the mass of a single volatile molecule.

Thermal desorption rate
~~~~~~~~~~~~~~~~~~~~~~~~~

Thermal desorption follows the Polanyi–Wigner law (Eq. 5):

.. math::

   R_\text{th} = 4\pi a^2\,N_s\,n\,m\,p_k\,\nu_i\,
   \exp\!\left(-\frac{T_\text{bind}}{T}\right),

where:

- *N*\ :sub:`s` = 1.5 × 10\ :sup:`15` cm\ :sup:`-2` is the surface site
  density (default).
- *m* is the mass of a single ice molecule.
- *p*\ :sub:`k` is the fractional surface coverage (see below).
- *ν*\ :sub:`i` is the characteristic oscillation frequency of the adsorbed
  molecule (Eq. 6):

.. math::

   \nu_i = \sqrt{\frac{2 N_s k_B T_\text{bind}}{\pi^2\,\mu\,m_H}},

where *μ* is the molecular weight and *m*\ :sub:`H` is the hydrogen atom mass.

- *T*\ :sub:`bind` is the binding temperature (850 K for CO, default).

Surface coverage
~~~~~~~~~~~~~~~~~

The fractional surface coverage interpolates between zeroth-order desorption
(monolayer coverage) and first-order desorption (sub-monolayer coverage)
(Eq. 7):

.. math::

   p_k = 1 - \exp\!\left(-\frac{\rho_i}{4\pi a^2 N_s n m}\right).

When the ice mass exceeds one monolayer, :math:`p_k \approx 1` and
desorption is zeroth-order (independent of coverage). Below one monolayer,
:math:`p_k \approx \rho_i / (4\pi a^2 N_s n m)` and desorption is
first-order.

Photodesorption rate
~~~~~~~~~~~~~~~~~~~~~

UV photons desorb ice molecules at a rate (Eq. 8):

.. math::

   R_\text{ph} = \pi a^2\,n\,m\,p_k\,(\gamma_\text{UV} + \gamma_\text{CR})\,Y,

where *Y* = 2.7 × 10\ :sup:`-3` is the photodesorption yield, and
*γ*\ :sub:`UV` and *γ*\ :sub:`CR` are the UV flux and the cosmic-ray-induced
UV flux, respectively. The cosmic-ray UV flux is (Eq. 9):

.. math::

   \gamma_\text{CR} = \eta_\text{CR}\,\sigma_\text{CR},

with *η*\ :sub:`CR` = 10\ :sup:`-17` s\ :sup:`-1` the cosmic-ray ionisation
rate. For the CO snow line, photodesorption is typically negligible.

.. note::

   The UV photodesorption flux attenuates with column depth into the disc.
   The photodesorption module computes the UV optical depth along the column
   from the disc surface.

Implicit solver
~~~~~~~~~~~~~~~~

The chemistry equations are **stiff**: adsorption and desorption rates can
be much faster than the dynamical time step. cuDisc uses a first-order
implicit iterative solver to update the ice and vapour densities at each cell
(Eqs. 10–11):

.. math::

   \rho_v^{n+1} &= \frac{\rho_v^n + \Delta t\,R_\text{des}^{n+1}}
                       {1 + \Delta t\,R_\text{ad}^{n+1}/\rho_v^{n+1}}, \\[4pt]
   \rho_i^{n+1} &= \rho_\text{tot} - \rho_v^{n+1},

where :math:`\rho_\text{tot} = \rho_v + \rho_i` is the total volatile mass
(conserved). Iteration continues until the mean fractional change is below
10\ :sup:`-5`.

Ice in the grain-size distribution
-------------------------------------

Ice is treated as a tracer carried by the non-ice dust grains. It is
distributed across grain-size bins in proportion to the ice mass fraction of
each bin (Eqs. 12–15). When coagulation occurs, the ice mass fraction is
transferred between bins according to the coagulation and fragmentation rates,
under the assumption that ice and non-ice are uniformly mixed within each
grain (rather than existing as a separate mantle on a refractory core).

Opacities with ice
-------------------

When ice is present, the dust opacity must account for the mixed ice–refractory
composition. cuDisc uses the Maxwell-Garnett effective medium theory to
compute composite grain opacities from the ice and dust volume fractions. The
optical constants of CO ice are from Gavdush et al. (2022), extended to
long wavelengths with Lorentz oscillator kernels and Kramers–Kronig relations.

Vapour remapping
-----------------

During the hydrostatic re-equilibration step, the gas density profile is
recomputed, which changes the column density in each cell. The vapour density
must be remapped consistently. This is done with a log-column interpolation
scheme (Eqs. 16–18) that conserves the total vapour column density in each
radial annulus.

Snow-line instability
----------------------

The CO snow line can undergo a thermal instability (analogous to the
Dwarf-Nova instability) when the opacity changes rapidly with temperature
near the condensation point. The instability criterion derived in Robinson et
al. (2026), following Pringle (1981), is

.. math::

   \frac{\mathrm{d}\log q_+}{\mathrm{d}\log T}
   > \frac{\mathrm{d}\log q_-}{\mathrm{d}\log T},

where *q*\ :sub:`+` and *q*\ :sub:`-` are the heating and cooling rates,
respectively. A simplified condition in terms of the ice column density is
(Eq. 26):

.. math::

   \frac{\mathrm{d}\log\Sigma_\text{ice}}{\mathrm{d}\log T} \lesssim -4.

cuDisc's 2D structure reveals that the instability is suppressed in
dynamically evolving discs because vertical volatile transport prevents
the limit-cycle behaviour seen in 1D models.

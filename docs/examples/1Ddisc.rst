.. _examples_1Ddisc:

1Ddisc — 1D viscous disc with photoevaporation
===============================================

``1Ddisc`` evolves a 1D vertically-integrated viscous disc with
photoevaporative mass loss. It is included as a fast reference case to
verify that the radial gas evolution in cuDisc agrees with established
1D disc models.

Source file: ``codes/1Ddisc.cpp``

Physical setup
--------------

The disc is evolved using the 1D diffusion equation for the gas surface
density Σ\ :sub:`g`:

.. math::

   \frac{\partial \Sigma_g}{\partial t}
   = \frac{3}{R}\frac{\partial}{\partial R}\!\left[
     \sqrt{R}\frac{\partial}{\partial R}\!\bigl(\nu\Sigma_g\sqrt{R}\bigr)
   \right]
   - \dot\Sigma_\text{wind}(R,t),

where the kinematic viscosity *ν* = *α c*\ :sub:`s` *H*\ :sub:`g` and
:math:`\dot\Sigma_\text{wind}` is the photoevaporative mass-loss rate.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Value
   * - Stellar mass
     - 1 M\ :sub:`☉`
   * - Initial disc mass
     - 0.05 M\ :sub:`☉`
   * - Disc inner radius
     - 0.1 au
   * - Disc outer radius
     - 200 au
   * - Radial cells
     - 300 (log-spaced)
   * - Turbulence α
     - 1 × 10\ :sup:`−3`
   * - Photoevaporation rate *Ṁ*\ :sub:`wind`
     - 10\ :sup:`−8` M\ :sub:`☉`/yr (constant)

How to run
----------

.. code-block:: bash

   make 1Ddisc
   mkdir -p output/1Ddisc
   ./1Ddisc

The run completes in seconds (no GPU required for the 1D case).

Expected results
----------------

The disc evolves on the viscous timescale (~Myr) until the photoevaporative
gap opens inside the hole radius and the disc clears from the inside out.
The gap-opening time and the final disc lifetime can be compared with
published 1D disc evolution models (e.g. Clarke, Gendrin & Sotomayor 2001;
Alexander et al. 2014).

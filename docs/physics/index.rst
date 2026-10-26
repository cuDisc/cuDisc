.. _physics:

Physics and Numerics
====================

This section describes the equations and algorithms that cuDisc solves. For
each module the physical motivation is given first, followed by the equations
and the numerical method used to solve them. Where possible, the relevant
equations from the code papers are cited by number.

A full description of the methods is given in:

- Robinson et al. (2024) — the cuDisc code paper (grid structure, dust
  dynamics, temperature solver, coagulation).
- Robinson, Owen & Booth (2026) — the ice–vapour chemistry paper.

.. toctree::
   :maxdepth: 1

   grid_structure
   dust_dynamics
   radiative_transfer
   coagulation
   ice_chemistry

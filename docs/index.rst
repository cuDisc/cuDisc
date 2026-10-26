.. cuDisc documentation master file

cuDisc Documentation
====================

**cuDisc** is a 2D, GPU-accelerated protoplanetary disc code for studying the
coupled evolution of dust dynamics, grain-size distributions, and disc
temperature structure. The code assumes axisymmetry and works in the
meridional (R, θ) plane, capturing both the radial and vertical structure of a
disc simultaneously.

The physical processes included in cuDisc are:

- **2D dust advection–diffusion** — a second-order finite-volume Godunov solver
  transports multiple grain-size populations through the disc.
- **Dust coagulation and fragmentation** — the Smoluchowski coagulation equation
  is solved to evolve a full grain-size distribution self-consistently.
- **Radiative transfer and temperature** — a hybrid ray-tracing plus
  multifrequency flux-limited diffusion (FLD) method computes the disc
  temperature in 2D.
- **1D gas evolution** — a viscous gas evolution equation with hydrostatic
  equilibrium closure evolves the gas component.
- **Ice–vapour chemistry** — adsorption, thermal desorption, and photodesorption
  of volatile ices (e.g. CO) are coupled to the dynamics.

cuDisc is written in C++17 with CUDA for GPU acceleration. A full description of
the numerical methods is given in the code papers listed in
:doc:`citation`.

.. note::

   You will need an NVIDIA GPU to run cuDisc. The code has been tested with
   CUDA 12.0 and later.

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Examples and Tests

   examples/index
   tests/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Physics and Numerics

   physics/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   citation
   changelog

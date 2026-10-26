.. _changelog:

Changelog
=========

All notable changes to cuDisc are documented here. The project follows
`Semantic Versioning <https://semver.org/>`_.

v2.0.0 (2026)
--------------

New features

- Ice–vapour chemistry module: adsorption, thermal desorption (Polanyi–Wigner),
  and UV photodesorption for volatile species (CO, H₂O).
- On-the-fly opacity update for mixed ice–refractory grains using Maxwell-Garnett
  effective medium theory.
- Vapour remapping scheme conserving column-integrated volatile mass during
  hydrostatic re-equilibration steps.
- Snow-line instability diagnostics.

Changes

- Ice-fraction field ``f_ice`` added to the dust primitive variable output
  (``dust<XXXX>.bin``).
- New binary files ``ice<XXXX>.bin`` for per-species ice mass density.

Bug fixes

- Fixed an off-by-one error in the ghost-cell filling for the polar upper
  boundary when ``Nghost > 2``.
- Corrected the Planck-mean opacity weighting in the multifrequency FLD
  solver at the first time step.

v1.0.0 (2024)
--------------

Initial public release accompanying Robinson, Booth & Owen (2024, MNRAS
529, 1524).

Features

- 2D axisymmetric dust dynamics: second-order Godunov scheme with Roe-averaged
  fluxes, implicit drag, and turbulent diffusion.
- Multifrequency flux-limited diffusion (FLD) for radiative transfer with
  ``n_bands`` frequency bands.
- Hybrid stellar irradiation: frequency-resolved ray-tracing with
  single-scattering correction.
- DSHARP opacities (Birnstiel et al. 2018) with bilinear log-log interpolation.
- Dust coagulation and fragmentation via the Smoluchowski equation with the
  BirnstielKernel and SimpleErosion model.
- Bogacki–Shampine 3(2) adaptive ODE integrator for coagulation.
- Logarithmic grain-size grid with ``SizeGrid``.
- Subdivided polar grid (``ThetaSpacing::subdiv``) for resolving disc
  atmosphere structure.
- Binary output compatible with the Python reader ``read_cudisc.py``.
- Four validation tests: 2D advection–diffusion, Golovin coagulation,
  DustPy comparison, Pinte et al. benchmark.
- Three example executables: ``steadyTD``, ``isoPD``, ``1Ddisc``.

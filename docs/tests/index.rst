.. _tests:

Tests
=====

cuDisc ships with four test problems that verify individual physical modules.
Each can be compiled with the corresponding ``make`` target.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Target
     - Description
   * - ``test_adv_diff``
     - Advection–diffusion of a 2D Gaussian dust pulse. Verifies second-order
       spatial convergence of the Godunov solver.
   * - ``test_coag_const``
     - Coagulation with the constant kernel, for which an analytic solution
       exists. Checks the Smoluchowski solver against the Golovin (1963)
       solution.
   * - ``test_coagdustpy``
     - Comparison of cuDisc's coagulation and fragmentation routines against
       the DustPy code for a 1D vertically integrated disc without radial
       drift.
   * - ``test_pinte_graindist``
     - 2D radiative-transfer benchmark in the spirit of Pinte et al. (2009).
       Checks the temperature structure computed by the FLD solver against a
       reference Monte Carlo solution.

Running a test
--------------

.. code-block:: bash

   make test_adv_diff
   ./test_adv_diff

Results are printed to the terminal and/or written to output files in
``codes/outputs/``. The Python scripts in ``tests/scripts/`` can be used to
reproduce the test figures from Robinson et al. (2024):

.. code-block:: bash

   cd tests/scripts
   python run_tests.py

The test scripts require the Python packages listed in ``requirements.txt``.

Test details
------------

.. rubric:: Advection–diffusion test (``test_adv_diff``)

A 2D Gaussian density distribution is advected and diffused on the cuDisc
grid. The L1 error norm is computed at a set of grid resolutions to confirm
second-order convergence. This test verifies both the transport and diffusion
parts of the dust dynamics solver described in :ref:`physics/dust_dynamics:Dust Dynamics`.

.. rubric:: Constant-kernel coagulation test (``test_coag_const``)

The Smoluchowski equation with the constant kernel :math:`K(m_1, m_2) = 1`
has the analytic solution (Golovin 1963)

.. math::

   n(m, t) = \frac{N_0}{\langle m \rangle^2}\,
   \left(\frac{t/t_c}{1 + t/t_c}\right)^{m/\langle m \rangle - 1}\,
   \frac{1}{1 + t/t_c}\,,

where :math:`t_c = 2/(N_0 K)` is the coagulation timescale. The test checks
that the numerical solution converges to this analytic result.

.. rubric:: DustPy comparison test (``test_coagdustpy``)

This test runs the coagulation and fragmentation solver without dust dynamics
(no radial drift or vertical diffusion) and compares the grain-size
distribution at fixed radii against an equivalent DustPy run. It is designed
to verify that cuDisc and DustPy agree on the steady-state grain-size
distribution, which is set by the balance between coagulation and
fragmentation.

.. rubric:: Radiative transfer benchmark (``test_pinte_graindist``)

Based on the benchmark problem of Pinte et al. (2009, A&A 498, 967), this
test computes the 2D temperature structure of an optically thick flared disc
with a single grain size. The result is compared to a reference solution
computed with RADMC-3D. It validates the stellar irradiation ray-tracing and
FLD routines described in :ref:`physics/radiative_transfer:Radiative Transfer`.

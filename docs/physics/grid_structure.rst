.. _physics_grid:

Grid Structure
==============

cuDisc uses a structured 2D mesh in the poloidal plane of an axisymmetric
protoplanetary disc. The coordinate system is cylindrical radius *R* and
polar angle *θ* measured from the midplane, so *θ = 0* corresponds to the
disc equatorial plane and increasing *θ* approaches the disc surface.

Coordinate system
-----------------

Cell interfaces are surfaces of constant *R* (cylinders) and constant *θ*
(cones). The cell (i, j) is bounded by radii *R*\ :sub:`e,i` and
*R*\ :sub:`e,i+1`, and by polar angles *θ*\ :sub:`e,j` and
*θ*\ :sub:`e,j+1`. In cylindrical coordinates (*R*, *Z*), the *Z*-faces are
therefore *not* flat planes but tilted surfaces.

.. note::

   This grid is a non-orthogonal quadrilateral mesh. The radial and vertical
   directions are not perpendicular to each other in the (*R*, *Z*) plane.
   The consequence is that the face normals in the *Z*-direction have a
   non-zero component in *R*.

Cell volumes and face areas
----------------------------

The volume of cell (i, j) is (Robinson et al. 2024, Eq. 1):

.. math::

   V_{ij} = \frac{1}{3}\,\Delta(R^3)_i\,\Delta(\tan\theta)_j

where :math:`\Delta(R^3)_i = R_{e,i+1}^3 - R_{e,i}^3` and
:math:`\Delta(\tan\theta)_j = \tan\theta_{e,j+1} - \tan\theta_{e,j}`. The
factor of 1/3 follows from integrating over the azimuthal angle and the
volume element :math:`R\,\mathrm{d}R\,\mathrm{d}Z`.

The face area in the *R*-direction (between cells (i−1, j) and (i, j)) is
(Eq. 2):

.. math::

   A^R_{ij} = R_{e,i}^2\,\Delta(\tan\theta)_j

The face area in the *Z*-direction (between cells (i, j−1) and (i, j)) is
(Eq. 3):

.. math::

   A^Z_{ij} = \frac{R_{e,i+1}^3 - R_{e,i}^3}{3\,R_{c,i}}\,
              \frac{1}{\cos\theta_{e,j}}

These forms are chosen to avoid numerical cancellation as *R* → 0.

Face normals
------------

The outward normal to the *R*-face pointing from cell (i−1, j) into cell
(i, j) is simply :math:`\hat{n}^R = (1, 0)` in cylindrical coordinates.

The outward normal to the *Z*-face pointing from cell (i, j−1) into cell
(i, j) is

.. math::

   \hat{n}^Z = \bigl(-\sin\theta_{e,j},\;\cos\theta_{e,j}\bigr)

reflecting the tilted nature of the constant-*θ* faces.

Grid spacing
------------

Radial grid spacing
~~~~~~~~~~~~~~~~~~~

The standard choice for disc simulations is logarithmic spacing, so that
cells have constant fractional width :math:`\Delta R / R`. This is set by
``RadialSpacing::log`` in the grid parameters.

For cases where higher resolution is needed at large radii, power-law spacing
with :math:`\Delta R \propto R^{p_R}` is available via
``RadialSpacing::power`` and the parameter ``R_power``.

Polar-angle grid spacing
~~~~~~~~~~~~~~~~~~~~~~~~~

The polar grid must resolve both the optically thin surface layers (which
require fine angular resolution near *θ* = 0) and the optically thick midplane
regions (which require cells that span a sufficient vertical column). The
**subdivided** grid (``ThetaSpacing::subdiv``) satisfies both requirements:

- From *θ* = 0 to ``theta_subdiv``, cells are uniformly spaced.
- From ``theta_subdiv`` to *θ*:sub:`max`, cells follow a power law with
  exponent ``theta_power``.

A typical choice is ``theta_subdiv = π/9`` (20°) and ``theta_power = 0.75``,
which concentrates cells in the disc atmosphere while still covering the
midplane.

Ghost cells
-----------

Two layers of ghost cells surround the active domain on each side (set by
``Nghost = 2`` in the grid parameters). Ghost cells are used exclusively to
implement boundary conditions; physical quantities are never evolved in them
directly. The solvers fill ghost cells at the start of each time step
according to the chosen boundary condition flags.

Boundary conditions
-------------------

Boundaries are specified per edge using flags from ``BoundaryFlags``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Edge
     - Default in examples
   * - Inner radial (*R*:sub:`min`)
     - Outflow (``open_R_inner``)
   * - Outer radial (*R*:sub:`max`)
     - Outflow (``open_R_outer``)
   * - Lower polar (*θ* = 0, midplane)
     - Closed/reflecting (midplane symmetry)
   * - Upper polar (*θ*:sub:`max`)
     - Outflow (``open_Z_outer``)

The midplane is always treated as a symmetry plane; no flag is needed for it.

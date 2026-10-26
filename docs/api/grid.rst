.. _api_grid:

Grid
====

Defined in ``headers/grid.h``.

Grid class
----------

The main grid class. Stores all geometric information for the 2D poloidal
mesh and provides accessor methods for cell centres, face positions,
volumes, and face areas.

.. code-block:: cpp

   class Grid;

Construction
~~~~~~~~~~~~

.. code-block:: cpp

   Grid(const Grid::params& p);

Construct a grid from a parameter struct. All geometric arrays are
allocated and filled on the GPU.

Grid::params
~~~~~~~~~~~~

.. code-block:: cpp

   struct Grid::params {
       int NR       = 0;         // Number of radial cells (active)
       int Nphi     = 0;         // Number of polar cells (active)
       int Nghost   = 2;         // Ghost cell layers (each side)
       double Rmin;              // Inner radial boundary [cm]
       double Rmax;              // Outer radial boundary [cm]
       double theta_min = 0;     // Lower polar boundary [rad] (midplane = 0)
       double theta_max;         // Upper polar boundary [rad]
       RadialSpacing  R_spacing     = RadialSpacing::linear;
       ThetaSpacing   theta_spacing = ThetaSpacing::linear;
       double theta_subdiv = 0;  // Subdivision angle for ThetaSpacing::subdiv
       double theta_power  = 1;  // Power-law exponent for ThetaSpacing::power
       double R_power      = 1;  // Power-law exponent for RadialSpacing::power
   };

Cell accessors
~~~~~~~~~~~~~~

The indices *i* and *j* run over the full domain including ghost cells:
``i`` ∈ [0, NR + 2·Nghost), ``j`` ∈ [0, Nphi + 2·Nghost).

.. code-block:: cpp

   // Cylindrical R of cell CENTRE (i) and cell EDGE (i)
   double Rc(int i) const;
   double Re(int i) const;

   // Cylindrical Z of cell centre (i,j) and lower Z-face (i,j)
   double Zc(int i, int j) const;
   double Ze(int i, int j) const;

   // Spherical r of cell centre (i,j) and lower r-face (i,j)
   double rc(int i, int j) const;
   double re(int i, int j) const;

   // Polar angle of cell centre (j) and lower polar face (j)
   double theta_c(int j) const;
   double theta_e(int j) const;

Volume and face areas
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Cell volume [cm^3]
   double volume(int i, int j) const;

   // R-face area between cells (i-1,j) and (i,j) [cm^2]
   double area_R(int i, int j) const;

   // Z-face area between cells (i,j-1) and (i,j) [cm^2]
   double area_Z(int i, int j) const;

Dimension members
~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   int NR;      // Active radial cells
   int Nphi;    // Active polar cells
   int Nghost;  // Ghost cell layers

Enumerations
-------------

RadialSpacing
~~~~~~~~~~~~~

.. code-block:: cpp

   enum class RadialSpacing {
       linear,   // Uniform ΔR
       log,      // Uniform Δ(log R), recommended for disc simulations
       power,    // ΔR ∝ R^R_power
   };

ThetaSpacing
~~~~~~~~~~~~

.. code-block:: cpp

   enum class ThetaSpacing {
       linear,   // Uniform Δθ
       subdiv,   // Uniform from 0→theta_subdiv, power law from theta_subdiv→theta_max
       power,    // Δθ ∝ θ^theta_power
   };

Staggering
~~~~~~~~~~

Used to specify staggered-grid face locations for fields.

.. code-block:: cpp

   enum class Staggering {
       none,   // Cell centres
       R,      // R-face centres
       phi,    // Polar-face centres
       both,   // Both faces (corner-centred)
   };

Physical constants (``headers/constants.h``)
--------------------------------------------

The following constants are available globally after including
``constants.h``:

.. code-block:: cpp

   constexpr double au      = 1.496e13;   // cm
   constexpr double Msun    = 1.989e33;   // g
   constexpr double Rsun    = 6.957e10;   // cm
   constexpr double Lsun    = 3.828e33;   // erg/s
   constexpr double year    = 3.156e7;    // s
   constexpr double GMsun   = 1.327e26;   // cm^3/s^2
   constexpr double k_B     = 1.381e-16;  // erg/K
   constexpr double m_H     = 1.673e-24;  // g
   constexpr double sigma_SB= 5.670e-5;   // erg/(cm^2 s K^4)
   constexpr double c_light = 2.998e10;   // cm/s

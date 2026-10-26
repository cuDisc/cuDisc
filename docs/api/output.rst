.. _api_output:

Output and Restart
==================

Defined in ``headers/write_output.h``.

write_file
----------

Write a snapshot of the dust and gas fields to binary files in a named
directory.

.. code-block:: cpp

   void write_file(const std::string&    dir,
                   int                   snap,
                   const Grid&           g,
                   const SizeGrid&       sizes,
                   const Field3D<Prims>& Ws_d,
                   const Field<Prims>&   Ws_g,
                   const CudaArray<double>& Sig_g,
                   const Field<double>&  T,
                   const Field<double>&  J);

This function writes three binary files per snapshot:

- ``<dir>/dust<XXXX>.bin`` — dust primitive variables for all species.
- ``<dir>/gas<XXXX>.bin`` — gas primitive variables.
- ``<dir>/rad<XXXX>.bin`` — radiation energy density *J* and temperature *T*.

where ``XXXX`` is the zero-padded snapshot index *snap*.

write_grids
-----------

Write grid metadata files. Call once before the time loop.

.. code-block:: cpp

   void write_grids(const std::string&      dir,
                    const Grid&             g,
                    const SizeGrid&         sizes,
                    const DSHARP_opacs&     opacs,
                    const WavelengthBinner& bins);

Files written:

- ``<dir>/grid.bin`` — cell centre and edge coordinates.
- ``<dir>/sizes.bin`` — grain-size bin radii and masses.
- ``<dir>/opacs.bin`` — opacity table values.
- ``<dir>/bins.bin`` — wavelength band boundaries.

read_active
-----------

Read a dust field from a binary restart snapshot.

.. code-block:: cpp

   void read_active(const std::string& dir,
                    int                snap,
                    const Grid&        g,
                    Field3D<Prims>&    Ws_d);

Binary file format
------------------

All binary files use the following layout:

1. Header: four 4-byte integers giving the array dimensions.
2. Data: IEEE 754 double-precision (8-byte) values in C row-major order.

The Python reader in ``codes/scripts/read_cudisc.py`` handles this format
transparently:

.. code-block:: python

   import read_cudisc as rc

   g = rc.Grid('output_dir')
   dust = rc.Dust('output_dir', snap=10)

   # Dust surface density of species k
   Sigma_d = dust.Sigma[k, :]

   # Midplane dust density
   rho_d_mid = dust.rho[k, :, 0]

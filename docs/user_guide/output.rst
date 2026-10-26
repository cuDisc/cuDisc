.. _user_guide_output:

Output and Analysis
===================

cuDisc writes output in a binary format for efficiency. A set of Python
utilities in ``codes/python/`` reads these files and provides tools for
visualisation and post-processing.

Writing output
--------------

At the start of every simulation, write the grid and opacity metadata once:

.. code-block:: cpp

   std::filesystem::path dir = "./codes/outputs/my_run";
   std::filesystem::create_directories(dir);

   write_grids(dir, &g, &sizes, &opacs, &bins);

Within the time loop, write a snapshot at each output time:

.. code-block:: cpp

   write_file(dir, Nout, g, Ws_d, Ws_g, Sig_g, T, J);
   Nout++;

The snapshot index ``Nout`` starts at 0 (the initial condition) and
increments by 1 for each output. The output times are written to a plain
text file for reference:

.. code-block:: cpp

   std::ofstream f_times(dir / "2Dtimes.txt");
   f_times << 0. << "\n";                    // initial snapshot
   for (double t_out : output_times)
       f_times << t_out << "\n";
   f_times.close();

Output file structure
~~~~~~~~~~~~~~~~~~~~~

For each snapshot, cuDisc writes three binary files:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``dustXXXX.bin``
     - Dust primitive variables (ρ, v\ :sub:`R`, v\ :sub:`φ`, v\ :sub:`Z`)
       for all grain sizes and grid cells
   * - ``gasXXXX.bin``
     - Gas primitive variables (ρ\ :sub:`g`, v\ :sub:`R`, v\ :sub:`Z`),
       surface density Σ\ :sub:`g`, and temperature *T*
   * - ``radXXXX.bin``
     - Multiband mean radiation intensity *J* (n_bands values per cell)

The ``XXXX`` suffix is the zero-padded snapshot index.

Grid and opacity metadata files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``write_grids`` call produces:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``grid.bin``
     - Cell-centre R and Z coordinates (cm)
   * - ``sizes.bin``
     - Grain radii and masses for each size bin
   * - ``opacs.bin``
     - Absorption and scattering opacities (n_spec × n_wavelengths)
   * - ``bins.bin``
     - Wavelength bin edges and centres

Restart files
~~~~~~~~~~~~~

Every 1000 time steps, cuDisc writes restart files so that a simulation can
be resumed if interrupted:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``restart_params.dat``
     - Scalar parameters: step count, time, CFL step, coagulation and
       temperature timers
   * - ``restart_dust.bin``
     - Dust primitive variable field
   * - ``restart_gas.bin``
     - Gas primitive variable and surface density field
   * - ``restart_T.bin``
     - Temperature field
   * - ``restart_J.bin``
     - Radiation field

To restart, cuDisc checks for the existence of ``restart_params.dat`` and
reads the restart fields if it is found:

.. code-block:: cpp

   std::ifstream f(dir / "restart_params.dat", std::ios::binary);
   if (f) {
       read_restart_file(dir / "restart_params.dat", count, t, dt_CFL,
                         t_coag, t_temp, dt_coag, dt_1perc, dummy);
       read_restart_quants(dir, Ws_d, Ws_g, Sig_g, T, J);
       dyn.read_active(g, dir);
       // recompute cs2, cs, D, nu from the restarted T and Sig_g
   }

Python analysis
----------------

The Jupyter notebook ``codes/python/disc_plotter.ipynb`` shows how to read
the binary output and make standard disc plots. Key functions provided in the
accompanying Python module include:

- Reading the grid metadata and constructing 2D (R, Z) coordinate arrays.
- Loading a snapshot's dust and gas fields.
- Computing surface densities from the 3D density field by vertical integration.
- Computing the grain-size distribution at a given radius.
- Plotting dust surface density maps, temperature maps, and size distributions.

.. tip::

   The Python analysis tools require NumPy, Matplotlib, and (optionally)
   SciPy. Install them with:

   .. code-block:: bash

      pip install -r requirements.txt

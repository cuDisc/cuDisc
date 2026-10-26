.. _user_guide_opacity:

Opacities
=========

cuDisc uses tabulated dust opacities. The default tables are the DSHARP
(Disk Structures Hosting Atomic and Molecular Protoplanetary Disks) opacities
of Birnstiel et al. (2018), which give absorption and scattering efficiencies
as a function of grain size and wavelength.

Loading an opacity table
-------------------------

Opacity tables are stored as plain-text files with columns for wavelength,
grain radius, absorption opacity *κ*:sub:`abs`, and scattering opacity
*κ*:sub:`sca`. A pre-computed DSHARP table is included in
``codes/opacities/``:

.. code-block:: cpp

   DSHARP_opacs opac_tab("./codes/opacities/dustkappa_DSHARP100.txt", false);

The second argument disables normalisation (leave it as ``false`` unless you
are providing a custom table in a different normalisation convention).

Interpolating onto the simulation grid
---------------------------------------

Opacities must be interpolated onto the simulation's grain-size and
wavelength grids:

.. code-block:: cpp

   int num_wavelengths = 100;   // number of wavelength bins for stellar heating

   DSHARP_opacs opacs(n_spec, num_wavelengths);

   opacs.generate_lam(1.e-1, 1.e5);   // wavelength grid: 0.1 to 1e5 microns
   opacs.generate_a(sizes);            // grain sizes from the SizeGrid
   opacs.interpolate_opacs(opac_tab);  // bilinear interpolation

Setting the gas opacity floor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, gas opacity is set to a fixed fraction of the smallest grain's
opacity. This prevents unphysically low opacity in regions where the dust
density is very small:

.. code-block:: cpp

   opacs.set_k_g_min_grain(1.e-12);  // gas opacity = 1e-12 * kappa_smallest_grain

Wavelength binning for FLD
---------------------------

The FLD temperature solver works on a coarser wavelength grid (``n_bands``)
than the stellar heating calculation (``num_wavelengths``). The
:cpp:class:`WavelengthBinner` groups the fine wavelength grid into bands:

.. code-block:: cpp

   int n_bands = 20;
   WavelengthBinner bins(num_wavelengths, opacs.lam(), n_bands);

Opacities and scattering fields are then binned before being passed to the
FLD solver:

.. code-block:: cpp

   Field3D<double> rhok_abs        = create_field3D<double>(g, num_wavelengths);
   Field3D<double> rhok_sca        = create_field3D<double>(g, num_wavelengths);
   Field3D<double> rhok_abs_binned = create_field3D<double>(g, n_bands);
   Field3D<double> rhok_sca_binned = create_field3D<double>(g, n_bands);

   // Fill rhok_abs and rhok_sca from current densities and opacities:
   calculate_total_rhokappa(g, Ws_d, Ws_g, rho_tot, opacs, rhok_abs, rhok_sca);

   // Bin for FLD:
   rhok_abs_binned = bins.bin_planck(g, rhok_abs, T);         // Planck-weighted
   bin_central(g, rhok_sca, rhok_sca_binned, num_wavelengths, n_bands); // geometric mean

Writing opacity grids to file
------------------------------

The grid layout, size distribution, and opacity tables are written to the
output directory at the start of each run so that the Python analysis tools
can read them:

.. code-block:: cpp

   write_grids(dir, &g, &sizes, &opacs, &bins);

Custom opacity tables
----------------------

To use a custom opacity table, provide a text file with the same column
format as the DSHARP table, then load it with the ``DSHARP_opacs``
constructor as above. The ``interpolate_opacs`` function performs bilinear
interpolation in log–log space.

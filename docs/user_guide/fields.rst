.. _user_guide_fields:

Fields and Data Storage
========================

cuDisc stores physical quantities in three container types: :cpp:class:`Field`,
:cpp:class:`Field3D`, and :cpp:class:`CudaArray`. All three allocate memory
on both the CPU (host) and the GPU (device) and transfer data between them
as needed.

``Field<T>`` — 2D cell-centred data
-------------------------------------

A :cpp:class:`Field\<T\>` holds one value per grid cell, indexed by (i, j)
where *i* is the radial index and *j* is the polar-angle index. It includes
ghost-cell layers.

Create a ``Field`` using the helper function :cpp:func:`create_field`:

.. code-block:: cpp

   Field<double> T    = create_field<double>(g);   // temperature
   Field<double> cs   = create_field<double>(g);   // sound speed
   Field<Prims>  Ws_g = create_field<Prims>(g);    // gas primitives

The template argument can be any trivially copyable type, including
user-defined structs such as :cpp:struct:`Prims`.

Staggered fields (defined at cell faces rather than centres) are created by
passing a :cpp:enum:`Staggering` argument:

.. code-block:: cpp

   Field<double> Fedge = create_field<double>(g, Staggering::R);   // face-centred in R
   Field<double> Fedge = create_field<double>(g, Staggering::phi); // face-centred in phi

Access is via the ``()`` operator:

.. code-block:: cpp

   T(i, j) = 100.;         // set cell (i,j)
   double t = T(i, j);     // read cell (i,j)

``Field3D<T>`` — 3D data (2D grid + size bins)
-------------------------------------------------

A :cpp:class:`Field3D\<T\>` adds a third axis of length ``Nd``, typically
used for the ``n_spec`` grain-size bins:

.. code-block:: cpp

   Field3D<Prims>  Ws_d = create_field3D<Prims>(g, n_spec);   // dust primitive vars
   Field3D<double> D    = create_field3D<double>(g, n_spec);  // diffusion coefficient

Access is via the ``()`` operator with three indices:

.. code-block:: cpp

   Ws_d(i, j, k).rho = 1.e-15;   // density for size bin k at cell (i,j)

The flat-array index is provided by ``Field3D<T>::index(i, j, k)`` for
situations where direct pointer arithmetic is needed.

``CudaArray<T>`` — 1D CUDA arrays
-----------------------------------

One-dimensional arrays that live on both host and device, used for
quantities that depend only on radius (e.g. gas surface density, kinematic
viscosity):

.. code-block:: cpp

   CudaArray<double> Sig_g = make_CudaArray<double>(g.NR + 2*g.Nghost);
   CudaArray<double> nu    = make_CudaArray<double>(g.NR + 2*g.Nghost);

Access is via the ``[]`` operator:

.. code-block:: cpp

   Sig_g[i] = 100.;   // g cm^-2

The ``Prims`` struct
---------------------

Primitive hydrodynamic variables for a single fluid species are stored in
the :cpp:struct:`Prims` struct defined in ``headers/dustdynamics.h``:

.. code-block:: cpp

   struct Prims {
       double rho;    // mass density (g cm^-3)
       double v_R;    // radial velocity (cm s^-1)
       double v_phi;  // azimuthal velocity (cm s^-1)
       double v_Z;    // vertical velocity (cm s^-1)
   };

All four components must be set for every cell and every species, including
ghost cells, before the first solver call.

Copying fields
--------------

Fields can be copied with :cpp:func:`copy_field`:

.. code-block:: cpp

   Field<double> T_old = create_field<double>(g);
   copy_field(g, T, T_old);   // deep copy T into T_old

This is used, for example, to save the temperature before a solver
iteration and compute the fractional change afterward.

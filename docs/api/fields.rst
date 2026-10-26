.. _api_fields:

Fields
======

Defined in ``headers/field.h`` and ``headers/cuda_array.h``.

Field<T>
--------

A 2D array of values of type *T* defined on a ``Grid``. Memory is allocated
on both the host (CPU) and the device (GPU) and can be synchronised in
either direction.

.. code-block:: cpp

   template <typename T>
   class Field;

Field<T> factory
~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <typename T>
   Field<T> create_field(const Grid& g,
                         Staggering stagger = Staggering::none);

Allocate a ``Field<T>`` sized for grid *g*. The total number of elements is
``(NR + 2*Nghost) × (Nphi + 2*Nghost)``. Memory is zero-initialised.

Field<T> element access
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   T& operator()(int i, int j);             // host access
   const T& operator()(int i, int j) const;

Indices follow the same ghost-cell convention as ``Grid``: *i* ∈ [0,
NR + 2·Nghost), *j* ∈ [0, Nphi + 2·Nghost). Active cells start at
``g.Nghost``.

Data transfer
~~~~~~~~~~~~~

.. code-block:: cpp

   void copy_host_to_device();
   void copy_device_to_host();

Field3D<T>
----------

A 3D array of type *T* for fields that have an additional species (grain-size)
dimension, such as the dust primitive variables ``Ws_d``.

.. code-block:: cpp

   template <typename T>
   class Field3D;

Field3D<T> factory
~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <typename T>
   Field3D<T> create_field3D(const Grid& g,
                              int n_species,
                              Staggering stagger = Staggering::none);

Field3D<T> element access
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   T& operator()(int k, int i, int j);             // k = species index
   const T& operator()(int k, int i, int j) const;

CudaArray<T>
------------

A 1D array allocated on both host and device, analogous to
``std::vector<T>`` but GPU-resident. Used for radially-varying quantities
such as the vertically-integrated gas surface density ``Sig_g``.

.. code-block:: cpp

   template <typename T>
   class CudaArray;

CudaArray<T> factory
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <typename T>
   CudaArray<T> make_CudaArray(int n);

Element access (host)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   T& operator[](int i);
   const T& operator[](int i) const;

Primitive variables struct (``Prims``)
--------------------------------------

The primitive variables for a single fluid species are packed into a struct:

.. code-block:: cpp

   struct Prims {
       double rho;    // Mass density [g/cm^3]
       double v_R;    // Radial velocity [cm/s]
       double v_phi;  // Azimuthal velocity [cm/s] (physical, not angular)
       double v_Z;    // Vertical velocity [cm/s]
   };

Field utilities
---------------

.. code-block:: cpp

   // Copy one field into another (both must have the same dimensions)
   template <typename T>
   void copy_field(const Field<T>& src, Field<T>& dst);

   template <typename T>
   void copy_field(const Field3D<T>& src, Field3D<T>& dst, int k_src, int k_dst);

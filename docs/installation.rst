.. _installation:

Installation
============

Prerequisites
-------------

cuDisc requires the following to compile and run:

- An **NVIDIA GPU** with CUDA Compute Capability 6.0 or higher.
- **CUDA Toolkit** 12.0 or later. The default path assumed by the makefile is
  ``/usr/local/cuda-12.0``; adjust this if your installation differs (see
  below).
- A C++17-capable compiler (GCC 9 or later is recommended). The CUDA toolkit
  ships ``nvcc`` which will drive the compilation.
- **Python 3** with NumPy and Matplotlib if you wish to use the analysis
  scripts and notebooks in ``codes/python/``. The full list of Python
  dependencies is in ``requirements.txt``.

Obtaining the code
------------------

Clone the repository from GitHub:

.. code-block:: bash

   git clone https://github.com/cuDisc/cuDisc.git
   cd cuDisc

Alternatively, download a tagged release from the
`Releases <https://github.com/cuDisc/cuDisc/releases>`_ page.

Configuring the makefile
------------------------

Open ``makefile`` in the root of the repository and set the ``CUDA_HOME``
variable to match your CUDA installation:

.. code-block:: make

   CUDA_HOME = /usr/local/cuda-12.0

If CUDA is installed elsewhere (e.g. via a module system on an HPC cluster)
use the path reported by ``which nvcc | sed 's|/bin/nvcc||'``.

Compiling the tests
-------------------

The fastest way to verify your build is to compile one of the included tests:

.. code-block:: bash

   make test_adv_diff

This builds the advection–diffusion test binary. The available test targets are
described in :ref:`tests/index:Tests`.

Compiling the example simulations
----------------------------------

The three example simulations are compiled individually:

.. code-block:: bash

   make steadyTD    # 2D dust dynamics + coagulation + radiative transfer
   make isoPD       # 2D isothermal disc (no temperature solver)
   make 1Ddisc      # 1D gas + two-population dust model

The resulting binaries are placed in the ``codes/outputs/`` directory. Each
example is described in :doc:`examples/index`.

Building cuDisc as a static library
-------------------------------------

cuDisc can also be compiled as a static library so that you can link your own
simulation codes against it:

.. code-block:: bash

   make lib

This produces ``lib/libcudisc.a``. A worked example of a ``makefile`` that
links against the library is provided in ``codes/makefile``. The library
approach is recommended when you are writing a custom simulation rather than
modifying one of the examples.

Installing the Python dependencies
------------------------------------

Install the Python packages needed for the analysis scripts:

.. code-block:: bash

   pip install -r requirements.txt

The ``codes/python/disc_plotter.ipynb`` notebook shows how to read and
visualise cuDisc output files once a simulation has been run.

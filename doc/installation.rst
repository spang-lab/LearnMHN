.. |br| raw:: html

    <br />

Installation
============

Install the *mhn* package using *pip*
-------------------------------------

Since the *mhn* package is largely written in `Cython <https://cython.org/>`_, a `Cython-supported C compiler <https://cython.readthedocs.io/en/latest/src/quickstart/install.html>`_,
such as GCC or MSVC, must be installed on your device before you can install this package. |br|
If that is the case, you can simply install the *mhn* package using *pip*:

.. code-block:: console

    pip install mhn

If the installation of *mhn* was successful, you should be able to import it with

.. code-block:: python

    import mhn

Use the CUDA implementation to accelerate computations
------------------------------------------------------

If your device has an Nvidia GPU, you can accelerate the computation of the
log-likelihood and its gradient for both the full and the restricted state space
with CUDA. For that you have to have CUDA and the CUDA compiler installed on your
device. You can check that in the terminal with

.. code-block:: console

    nvcc --version

If this command is recognized, then CUDA should be installed on your device.
During installation the package will automatically check if the CUDA compiler
is installed on your device and will enable the corresponding functions if this is the case.
While running a Python script you can test if the *mhn* package has access to GPU-accelerated
functions using the :code:`cuda_available()` function as shown below:

.. code-block:: python

    import mhn

    print(mhn.cuda_available())

    # the three possible results are also available as constants:
    # CUDA_AVAILABLE, CUDA_NOT_AVAILABLE, CUDA_NOT_FUNCTIONAL

    if mhn.cuda_available() == mhn.CUDA_AVAILABLE:
        print('CUDA is available')

    if mhn.cuda_available() == mhn.CUDA_NOT_AVAILABLE:
        print('CUDA compiler nvcc was not present during installation')

    if mhn.cuda_available() == mhn.CUDA_NOT_FUNCTIONAL:
        print('CUDA compiler nvcc available but CUDA functions not working. Check CUDA installation')


Pay special attention to the :code:`CUDA_NOT_FUNCTIONAL` case. This indicates that while
the CUDA compiler is installed, basic functionalities like GPU memory allocation
are not working as expected. This likely points to an issue with your CUDA drivers,
so you should verify your CUDA installation.

If you cannot resolve :code:`CUDA_NOT_FUNCTIONAL`  by changing CUDA drivers, we recommend to install the package with CPU support only.
This can be accomplished on Linux via

.. code-block:: console

    export INSTALL_MHN_NO_CUDA=1
    pip install mhn

and on Windows via

.. code-block:: console

    set INSTALL_MHN_NO_CUDA=1
    pip install mhn


If you installed *nvcc* after installing the *mhn* package, you have to reinstall this package to gain access to the CUDA functions.

Reinstalling the package for CUDA-related reasons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to reinstall the package because you want to either
enable or disable CUDA support, you should add the :code:`--no-cache-dir` flag during
installation to ensure that *pip* does not use a cached version of the
package and that the package is actually recompiled:

.. code-block:: console

    pip uninstall mhn
    pip install mhn --no-cache-dir

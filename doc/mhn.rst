A detailed description of *mhn*'s subpackages, submodules and functions
=======================================================================

This part of the documentation contains descriptions of all functions included in
the *mhn* package.

Directly callable functions and constants
-----------------------------------------

.. autofunction:: mhn.set_seed

.. autofunction:: mhn.cuda_available


.. data:: mhn.CUDA_AVAILABLE

    Constant returned by :func:`cuda_available` if CUDA functions are available and working correctly.

.. data:: mhn.CUDA_NOT_AVAILABLE

    Constant returned by :func:`cuda_available` if the CUDA compiler (`nvcc`) was not present during installation.
    If this was not expected, ensure that the CUDA toolkit is properly installed and accessible.

.. data:: mhn.CUDA_NOT_FUNCTIONAL

    Constant returned by :func:`cuda_available` if the CUDA compiler (`nvcc`) is available,
    but CUDA functions are not working as expected.
    In this case, check your CUDA drivers and installation for potential issues.

.. raw:: html

   <br>
   <hr>

The *full_state_space* subpackage
---------------------------------

.. automodule:: mhn.full_state_space
    :noindex:

.. toctree::
    :maxdepth: 1

    View documentation of full_state_space <full_state_space/mhn.full_state_space>

The *training* subpackage
-------------------------

.. automodule:: mhn.training
    :noindex:

.. toctree::
    :maxdepth: 1

    View documentation of training <training/mhn.training>

The *optimizers* submodule
--------------------------

.. automodule:: mhn.optimizers
    :noindex:

.. toctree::
    :maxdepth: 1

    View documentation of optimizers <optimizers>

The *model* submodule
---------------------

.. automodule:: mhn.model
    :noindex:

.. toctree::
    :maxdepth: 1

    View documentation of model <model>

The *utilities* submodule
-------------------------

.. automodule:: mhn.utilities
    :noindex:

.. toctree::
    :maxdepth: 1

    View documentation of utilities <utilities>


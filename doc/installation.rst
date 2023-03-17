
Installation
============

You can install the *mhn* package using pip:

.. code-block:: console

    pip3 install mhn

After completing the installation of this package you should be able to import it by calling

.. code-block:: python

    import mhn

Using the CUDA implementation to accelerate score computations
--------------------------------------------------------------

If your device has an Nvidia GPU, you can accelerate the computation of the log-likelihood score and its gradient for both the full and the restricted state-space with CUDA. For that you have to have CUDA and the CUDA compiler installed on your device. You can check that in the terminal with

.. code-block:: console

    nvcc --version

If this command is recognized, then CUDA should be installed on your device.
During installation the package will automatically check if your the CUDA compiler
is installed and will enable the corresponding functions if this is the case.
While running a Python script you can test if the *mhn* package has access to GPU-accelerated
functions using the :code:`state_space_restriction` submodule as shown below:

.. code-block:: python

    from mhn.ssr import state_space_restriction

    print(state_space_restriction.cuda_available())

    # the three possible results are also available as constants:
    # CUDA_AVAILABLE, CUDA_NOT_AVAILABLE, CUDA_NOT_FUNCTIONAL

    if state_space_restriction.cuda_available() == state_space_restriction.CUDA_AVAILABLE:
        print('CUDA is available')

    if state_space_restriction.cuda_available() == state_space_restriction.CUDA_NOT_AVAILABLE:
        print('CUDA compiler nvcc was not present during installation')

    if state_space_restriction.cuda_available() == state_space_restriction.CUDA_NOT_FUNCTIONAL:
        print('CUDA compiler nvcc available but CUDA functions not working. Check CUDA installation')

Be especially aware of the :code:`CUDA_NOT_FUNCTIONAL` case: This means that the CUDA compiler is installed on your device but basic functionalities like allocating memory on the GPU are not working as expected. In this case something is probably wrong with your CUDA drivers and you should check your CUDA installation.

If you installed *nvcc* after installing the *mhn* package, you have to reinstall this package to gain access to the CUDA functions.
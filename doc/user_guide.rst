.. |br| raw:: html

    <br />


How to train your MHN
=====================

If you want to learn a new MHN from some mutation data, *mhn*'s :code:`optimizers` submodule
is probably the place you are looking for. It contains *Optimizer* classes for data
with and without age information for the individual samples.

Learn an MHN from data with no age information
----------------------------------------------

You can learn a new MHN from data with no age information for the individual samples
with the :code:`StateSpaceOptimizer` class.

.. code-block:: python

    from mhn.optimizers import StateSpaceOptimizer
    opt = StateSpaceOptimizer()


We can specify the data that we want our MHN to be trained on:

.. code-block:: python

    opt.load_data_matrix(data_matrix)

Here, :code:`data_matrix` can either be a *numpy* matrix or a *pandas* DataFrame, in which rows represent samples and
columns represent events.
If it is a *numpy* matrix, then you should set :code:`dtype=np.int32`, else you might get
a warning. |br|
Alternatively, if your training data is stored in a CSV file, you can call

.. code-block:: python

    opt.load_data_from_csv(filename, delimiter)

where :code:`delimiter` is the delimiter separating the items in the CSV file (default: :code:`,`).
Internally, this method uses *pandas*' :code:`read_csv()` function to extract the data from the CSV file.
All additional keyword arguments given to this method will be passed on to that *pandas* function (see `read_csv() <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_).
This means parameters like :code:`usecols` or :code:`skiprows` of the :code:`read_csv()` function
can also be used as parameters for this method:

.. code-block:: python

    # loads data from a CSV file, but does not include rows 0 and 10
    opt.load_data_from_csv(filename, delimiter, skiprows=[0, 10])


You can access the loaded data matrix with

.. code-block:: python

    loaded_matrix = opt.training_data

If you work with a CUDA-capable device, you can choose which device you want to use to train a new MHN:

.. code-block:: python

    # uses both CPU and GPU depending on the number of mutations in the individual sample (default)
    opt.set_device(StateSpaceOptimizer.Device.AUTO)

    # use the CPU to compute log-likelihood score and gradient
    opt.set_device(StateSpaceOptimizer.Device.CPU)

    # use the GPU to compute log-likelihood score and gradient
    opt.set_device(StateSpaceOptimizer.Device.GPU)

    # you can also access the Device enum directly with an Optimizer object
    opt.set_device(opt.Device.AUTO)

You could also change the initial theta that is the starting point for training, which by default is the independence model
used by Schill et al. (2019), with

.. code-block:: python

    opt.set_init_theta(init_theta)

If you want to regularly save the progress during training you can use the :code:`save_progress()` method:

.. code-block:: python

    # in this example we want to make a backup every 100 iterations
    steps = 100
    # we want to overwrite the previous backup file
    always_new_file = False
    # we want our backup file to be named 'mhn_training_backup.npy'
    filename = 'mhn_training_backup.npy'

    opt.save_progress(steps=steps, always_new_file=always_new_file, filename=filename)

You can also specify a callback function that is called after each training step:

.. code-block:: python

    # In this example we create a callback function that prints
    # the current theta matrix after each training step.
    # Make sure that your callback function takes the theta matrix as parameter
    # else you will get an error.
    def our_callback_function(theta: np.ndarray):
        print(theta)

    opt.set_callback_func(our_callback_function)

Finally, you can train a new MHN with

.. code-block:: python

    opt.train(
        lam=1/500,          # the lambda value used for L1 regularization
        maxit=5000,         # the maximum number of training iterations
        round_result=True,  # round the resulting theta matrix to two decimal places
    )

This function will return an :code:`MHN` object (see :ref:`here <*model*: A submodule containing the MHN classes>`) that contains the learned model. |br|
You can also access the learned model via the :code:`result` property:

.. code-block:: python

    learned_mhn = opt.result

The documentation of the :code:`StateSpaceOptimizer` can be found :ref:`here <Available Optimizers in the *optimizers* module>`.


Learn an MHN from data with age information
-------------------------------------------

You can learn a new MHN from data with age information with the :code:`DUAOptimizer` class.

.. code-block:: python

    from mhn.optimizers import DUAOptimizer
    opt = DUAOptimizer()

We can specify the data that we want our MHN to be trained on:

.. code-block:: python

    opt.load_data(data_matrix, age_array)

Here, :code:`data_matrix` *has* to be a *numpy* matrix, which should have :code:`dtype=np.int32` and :code:`age_array`
has to be a *numpy* array with :code:`dtype=np.double`. |br|
Except for methods that load data like :code:`load_data_from_csv()`, the :code:`DUAOptimizer` class supports all methods
described in the :ref:`previous section <Learn an MHN from data with no age information>`. |br|
The documentation of the :code:`DUAOptimizer` can also be found :ref:`here <Available Optimizers in the *optimizers* module>`.
.. |br| raw:: html

    <br />


How to train your MHN
=====================

If you want to learn a new MHN from mutation data, the :code:`optimizers` submodule
is likely where you should start. It currently contains *Optimizer* classes for training
a *classical* MHN (cMHN) (see `Schill et al. (2020) <https://academic.oup.com/bioinformatics/article/36/1/241/5524604>`_)
or an *observation* MHN (oMHN) (see `Schill et al. (2024) <https://link.springer.com/chapter/10.1007/978-1-0716-3989-4_14>`_).

Configure the Optimizer
-----------------------

You can learn a new MHN from cross-sectional data with the :code:`Optimizer` class:

.. code-block:: python

    from mhn.optimizers import Optimizer
    opt = Optimizer()

By default, this class will train the most recent type of MHN. To train an older type,
you can specify it explicitly:

.. code-block:: python

    # Example: training a classical MHN (cMHN) that does not account for the collider bias
    opt = Optimizer(Optimizer.MHNType.cMHN)

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
    opt.set_device(Optimizer.Device.AUTO)

    # use the CPU to compute log-likelihood and gradient
    opt.set_device(Optimizer.Device.CPU)

    # use the GPU to compute log-likelihood and gradient
    opt.set_device(Optimizer.Device.GPU)

    # you can also access the Device enum directly with an Optimizer object
    opt.set_device(opt.Device.AUTO)

You could also change the initial theta that is the starting point for training, which by default is the independence model
used by Schill et al. (2019), with

.. code-block:: python

    opt.set_init_theta(init_theta)

If you want to regularly save the progress during training, you can use the :code:`save_progress()` method:

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

    # In this example, we create a callback function that prints
    # the current theta matrix after each training step.
    # Ensure that your callback function accepts the theta matrix as a parameter;
    # otherwise, it will raise an error.
    def our_callback_function(theta: np.ndarray):
        print(theta)

    opt.set_callback_func(our_callback_function)

During training, a regularization penalty is applied to prevent overfitting. The
:code:`Optimizer` class currently supports three types: the L1-penalty (used by default), the L2-penalty, and
a custom symmetrical penalty that is further discussed in `Schill et al. (2024) <https://link.springer.com/chapter/10.1007/978-1-0716-3989-4_14>`_. |br|
The following code snippet shows how to set a penalty:

.. code-block:: python

     # for the L1-penalty, we set
     opt.set_penalty(opt.Penalty.L1)
     # for the L2-penalty, we set
     opt.set_penalty(opt.Penalty.L2)
     # for the symmetrical penalty, we set
     opt.set_penalty(opt.Penalty.SYM_SPARSE)

Train a new MHN model
---------------------

Once your optimizer is configured, you can call the :code:`lambda_from_cv()` method
to find the best penalty strength ("lambda") for training by doing cross-validation. |br|
The :code:`lambda_from_cv()` method takes either a sequence of lambdas that should be tested or
the minimum, maximum and step size for potential lambda values. In the latter case,
the method will create a range of possible lambdas with logarithmic grid-spacing,
e.g. :code:`(0.0001, 0.0010, 0.0100, 0.1000)` for :code:`lambda_min=0.0001`,
:code:`lambda_max=0.1` and :code:`steps=4`. |br|
In this example, we opted for the latter option:

.. code-block:: python

    import mhn
    # use a seed to make the cross-validation results reproducible
    mhn.set_seed(0)

    cv_lambda = opt.lambda_from_cv(
        lambda_min=1e-4,       # the smallest lambda value evaluated
        lambda_max=1e-1,       # the largest lambda value evaluated
        steps=4,               # total number of lambda values evaluated
        nfolds=5,              # number of cross-validation folds
        show_progressbar=True  # show a progressbar during cross-validation
    )

Finally, you can train a new MHN with

.. code-block:: python

    opt.train(
        lam=cv_lambda,      # the lambda value used for regularization
        maxit=5000,         # the maximum number of training iterations
        round_result=True,  # round the resulting theta matrix to two decimal places
    )

This function will return an :code:`MHN` object (see :ref:`here <*model*: A submodule containing the MHN classes>`) that contains the learned model. |br|
You can also access the learned model via the :code:`result` property:

.. code-block:: python

    learned_mhn = opt.result

The documentation of all available optimizer classes can be found :ref:`here <Available Optimizers in the *optimizers* module>`.

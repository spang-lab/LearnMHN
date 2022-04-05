# Implementing a learning algorithm for Mutual Hazard Networks

We implement an algorithm to approximate the gradients for training a MHN as described
in '[Scaling up Continuous-Time Markov Chains Helps Resolve Underspecification](https://arxiv.org/pdf/2107.02911.pdf)' 
by Gotovos et al. (2021) and compare the approximated gradients with the exact ones.  
For this we also implement the State Space Restriction, a similar method to compute the
exact gradient more efficiently.

## Installation

Before you run the Python scripts in this project you have to install some third party
packages:  

```bash
pip3 install -r src/requirements.txt
```

As this project relies on Cython code, you have to compile it as well with

```bash
python3 setup.py build_ext --inplace [--force]
```

The ``--force`` flag is optional and only needed if you want to force the script
to recompile the Cython code. This would for example be necessary if you later install 
the CUDA compiler ``nvcc`` and wanted to recompile the Cython code to get access to
the CUDA implementation of the State Space Restriction.  
Note: The Cython implementation of State Space Restriction might not work correctly if you used
``conda`` to install the Python packages. The reason might be that the Cython code 
uses BLAS operations that are provided by scipy and that the scipy package of ``conda`` 
uses a different BLAS implementation than the standard scipy package from pip.

## The script to reproduce our results ("main script")

To reproduce the results shown on the poster you can call
```bash
python3 compare_methods.py
```

This script generates the gradient error plot and the plot to compare the runtime 
of the approximated gradient with the State Space Restriction.


## The Cython modules

Even though there is an Python implementation for all algorithms used in this project,
it is highly recommended to use the Cython implementations as they are much faster.
To use them you simply have to import the corresponding package into your Python script:  
For the State Space Restriction

```python
import StateSpaceRestrictionCython
```

For the approximated gradient by Gotovos et al.:
```python
import approximate_gradient_cython
```

The functions that compute the gradients in both Cython modules need the mutation data
to be wrapped into an State_storage class object. To do that you first have to import
the corresponding Cython module:

```python
import state_storage
```

You can then convert the binary numpy matrix that represents the mutation data with

```python
# numpy_mutation_data is a binary numpy matrix
# the rows represent tumor samples, the columns genes
mutation_data = state_storage.State_storage(numpy_mutation_data)
```
The wrapped mutation data can then be used to compute the score and gradient of the current MHN
, e.g. using State Space Restriction:
```python
gradient, score = StateSpaceRestrictionCython.gradient_and_score(theta, mutation_data)
```
or using the approximation by Gotovos et al.:
```python
burn_in_sample_size = 10
sampled_paths_num = 50
gradient, score = approximate_gradient_cython.gradient_and_score_using_c(np.exp(theta), mutation_data, sampled_paths_num, burn_in_sample_size)
```

## Using the CUDA implementation for State Space Restriction
If you have a Nvidia GPU, you can accelerate the computation of the gradient for 
State Space Restriction with CUDA. For that you have to have CUDA and the CUDA compiler
installed on your device. You can check that in the terminal with
```bash
nvcc --version
```
If this command is recognized, then CUDA should be installed on your device.  
During the compilation of the Cython modules the CUDA code is also compiled if 
``nvcc`` is available. If you install ``nvcc`` after you already compiled the
Cython code, recompile everything using the ``--force`` flag.

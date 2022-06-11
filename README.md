# LearnMHN

Main implementation of the algorithm to learn an MHN from data

## Install the mhn package

You can install the mhn package using pip:

```bash
pip3 install -e /path/to/this/directory
```

After completing the installation of this package you should be able to import it by calling
```python
import mhn
```

If a new version of the mhn package is available, you can upgrade your installation with
```bash
pip3 install -e /path/to/this/directory --upgrade
```

## A quick overview

The package contains the original MHN functions implemented in Python. You import them from ``mhn.original``:
```python
from mhn.original import Likelihood, ModelConstruction, RegularizedOptimization, UtilityFunctions
```
It also contains functions to compute the Fisher information for a given MHN and use
Natural Gradient Descent to train a new model. You can use those functions by importing
```python
from mhn.original import FisherFunctions
```
Lastly, you can train a MHN using state-space restriction. The corresponding functions
can be imported with
```python
from mhn.ssr import state_space_restriction, state_storage
```

## Using the CUDA implementation of State-Space Restriction
If your device has a Nvidia GPU, you can accelerate the computation of the gradient and score for
State Space Restriction with CUDA. 
For that you have to have CUDA and the CUDA compiler
installed on your device. You can check that in the terminal with
```bash
nvcc --version
```
If this command is recognized, then CUDA should be installed on your device.  
You can also use the following function of the ```state_space_restriction``` submodule:
```python
from mhn.ssr import state_space_restriction

print(state_space_restriction.cuda_available())

# the three possible results are also available as constants:
# CUDA_AVAILABLE, CUDA_NOT_AVAILABLE, CUDA_NOT_FUNCTIONAL

if state_space_restriction.cuda_available() == state_space_restriction.CUDA_AVAILABLE:
    print('CUDA is available')

if state_space_restriction.cuda_available() == state_space_restriction.CUDA_NOT_AVAILABLE:
    print('The CUDA compiler nvcc could not be found')

if state_space_restriction.cuda_available() == state_space_restriction.CUDA_NOT_FUNCTIONAL:
    print('CUDA compiler nvcc available but CUDA functions not working. Check CUDA installation')
```

Be especially aware of the ```CUDA_NOT_FUNCTIONAL``` case: Even though CUDA
is not functional, the CUDA functions will run with no error, but will
return wrong results. In this case
something is probably wrong with your CUDA drivers and you should check your CUDA
installation.  
If you install ``nvcc`` after installing the ``mhn`` package, you have to
run 
```bash
pip3 install -e /path/to/this/directory --upgrade
```
to use the CUDA functions of this package.

## How to train a new MHN

The simplest way to train a new MHN is to import the ```optimizers``` module and
use the ```StateSpaceOptimizer``` class.
```python
from mhn.optimizers import StateSpaceOptimizer
opt = StateSpaceOptimizer()
```
We can specify the data that we want our MHN to be trained on:
```python
opt = opt.load_data_matrix(data_matrix)
```
Make sure, that the binary numpy matrix ```data_matrix``` is set to ```dtype=np.int32```, else you 
might get an error. Alternatively, if your training data is stored in a CSV file, you can call
```python
opt = opt.load_data_from_csv(filename, delimiter)
```
where ```delimiter``` is the delimiter separating the items in the CSV file (default: ``';'``). If
the CSV file contains more than just the binary matrix, e.g. the gene names or 
the sample names, you can use the optional 
arguments ```first_row, last_row, first_col, last_col``` to specify the range of
rows and columns, which contain the actual binary matrix without anything else.
If you do not do that, you will likely get wrong results.  
If you want to make sure that the matrix was loaded correctly, you can get 
the loaded matrix with
```python
loaded_matrix = opt.bin_datamatrix
```
By default, the optimizer will use the regularized score and gradient using 
state-space restriction as defined in ```mhn/ssr/learnMHN```. If you want to
use a different score and gradient function, you can change that with the method
```python
opt = opt.set_score_and_gradient_function(score_func, gradient_func)
```
You could also change the initial theta that is the starting point for training, which by default
is an independence model, with
```python
opt = opt.set_init_theta(init_theta)
```
If you want to regularly save the progress during training you can use
```python
opt = opt.save_progress(steps=-1, always_new_file=False, filename='theta_backup.npy')
```
The parameters of this method are  
``steps`` (default: ``-1``): if positive, the number of iterations between two progress storages  
``always_new_file`` (default: ``False``): if True, creates a new file for every progress storage, 
else the former progress file is overwritten each time  
``filename`` (default: ``"theta_backup.npy"``): the file name of the progress file.

Lastly, you could specify a callback function that is called after each training step
```python
def some_callback_function(theta: np.ndarray):
    pass

opt = opt.set_callback_func(some_callback_function)
```

Finally, you can train a new MHN with
```python
from mhn.optimizers import StateSpaceOptimizer
opt = StateSpaceOptimizer()
opt = opt.load_data_from_csv(filename, delimiter)
opt.train()
```
Some important parameters of the ``train`` method include  
``lam`` (default: ``0``), which is
a tuning parameter to control regularization,  
``maxit`` (default: ``5000``), which is the maximum
number of training iterations,  
```reltol``` (default: ``1e-7``), which is the gradient norm at which the training terminates and  
```round_result``` (default: ``True``), which, if set to True, rounds the result to two decimal places  
  
The resulting MHN is returned by the ```train()``` method, but can also be obtained
from the ```result``` parameter:
```python
new_mhn = opt.result
```
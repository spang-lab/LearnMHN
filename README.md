# *mhn*: A Python Package to Efficiently Compute Mutual Hazard Networks

Mutual Hazard Networks (MHN) were first introduced by [Schill et al. (2019)](https://academic.oup.com/bioinformatics/article/36/1/241/5524604)
and are used to model cancer progression.  
This Python package can be used to work with MHNs. It includes functions that were part of the
original R implementation as well as functions that make use of state-space restriction 
to make learning a new MHN from cancer data faster and more efficient.   
There are optimizer classes for data with known sample ages as well as for data without, which make learning a new MHN possible with
only a few lines of code.  

## Documentation

A detailed documentation of the *mhn* package is available [here](https://learnmhn.readthedocs.io/en/latest/index.html).

## Install the mhn Package

You can install the mhn package using pip:

```bash
pip install mhn
```

After completing the installation of this package you should be able to import it by calling
```python
import mhn
```

If a new version of the mhn package is available, you can upgrade your installation with
```bash
pip install --upgrade mhn
```

## A Quick Overview

The package contains the original MHN functions implemented in Python. You import them from ``mhn.original``:

```python
from mhn.full_state_space import Likelihood, ModelConstruction, RegularizedOptimization, UtilityFunctions
```
You can train an MHN using state-space restriction. The corresponding functions
can be imported with
```python
from mhn.ssr import state_space_restriction, state_containers
```
Training a new MHN can be as simple as writing the following few lines of code:

```python
from mhn.optimizers import cMHNOptimizer

opt = cMHNOptimizer()
opt = opt.load_data_from_csv("path/to/training_data")
new_mhn = opt.train()
new_mhn.save("path/to/saving/location")
```
We will look at the methods of the Optimizer class in more detail below.

## Using the CUDA Implementation to Accelerate Score Computations
If your device has an Nvidia GPU, you can accelerate the computation of the log-likelihood score and its gradient for
both the full and the restricted state-space with CUDA. 
For that you have to have CUDA and the CUDA compiler
installed on your device. You can check that in the terminal with
```bash
nvcc --version
```
If this command is recognized, then CUDA should be installed on your device.  
You can also use the following function to test if the *mhn* package has access to 
GPU-accelerated  functions:
```python
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
```

Be especially aware of the ```CUDA_NOT_FUNCTIONAL``` case: This means that the CUDA compiler
is installed on your device but basic functionalities like allocating memory on the GPU
are not working as expected.
In this case something is probably wrong with your CUDA drivers and you should check your CUDA
installation.

If you cannot resolve ```CUDA_NOT_FUNCTIONAL``` by changing CUDA drivers, we recommend to install the package with CPU support only.
This can be accomplished on Linux via
```bash
export INSTALL_MHN_NO_CUDA=1
pip install mhn
```
and on Windows via
```bash
set INSTALL_MHN_NO_CUDA=1
pip install mhn
```


If you installed ``nvcc`` after installing the ``mhn`` package, you have to
reinstall this package to gain access to the CUDA functions.

### Reinstalling the Package for CUDA-Related Reasons

If you want to reinstall the package because you want to either 
enable or disable CUDA support, you should add the ```--no-cache-dir``` flag during 
installation to ensure that *pip* does not use a cached version of the 
package and that the package is actually recompiled:

```bash
pip uninstall mhn
pip install mhn --no-cache-dir
```

## How to Train a New MHN

The simplest way to train a new MHN is to import the ```optimizers``` module and
use the ```cMHNOptimizer``` class.

```python
from mhn.optimizers import cMHNOptimizer

opt = cMHNOptimizer()
```
We can specify the data that we want our MHN to be trained on:
```python
opt.load_data_matrix(data_matrix)
```
Make sure, that the binary numpy matrix ```data_matrix``` is set to ```dtype=np.int32```, else you 
might get an error. Alternatively, if your training data is stored in a CSV file, you can call
```python
opt.load_data_from_csv(filename, delimiter)
```
where ```delimiter``` is the delimiter separating the items in the CSV file (default: ``','``). 
Internally, this method uses pandas' ```read_csv()``` function to extract the data from the CSV file.
All additional keyword arguments given to this method will be passed on to that
pandas function. This means parameters like ```usecols``` or ```skiprows``` of the ```read_csv()```
function can also be used as parameters for this method.  
If you want to make sure that the matrix was loaded correctly, you can get 
the loaded matrix with

```python
loaded_matrix = opt.training_data
```
If you work with a CUDA-capable device, you can choose which device you want to use to 
train a new MHN:
```python
# uses both CPU and GPU depending on the number of mutations in the individual sample
opt.set_device(cMHNOptimizer.Device.AUTO)
# use the CPU to compute log-likelihood score and gradient
opt.set_device(cMHNOptimizer.Device.CPU)
# use the GPU to compute log-likelihood score and gradient
opt.set_device(cMHNOptimizer.Device.GPU)
# you can also access the Device enum directly with an Optimizer object
opt.set_device(opt.Device.AUTO)
```
You could also change the initial theta that is the starting point for training, which by default
is an independence model, with
```python
opt.set_init_theta(init_theta)
```
If you want to regularly save the progress during training you can use
```python
opt.save_progress(steps=-1, always_new_file=False, filename='theta_backup.npy')
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

opt.set_callback_func(some_callback_function)
```

Finally, you can train a new MHN with

```python
from mhn.optimizers import cMHNOptimizer

opt = cMHNOptimizer()
opt = opt.load_data_from_csv(filename, delimiter)
opt.train()
```
Some important parameters of the ``train`` method include  
``lam`` (default: ``0``), which is
the lambda tuning parameter to control L1 regularization,  
``maxit`` (default: ``5000``), which is the maximum
number of training iterations,  
```reltol``` (default: ``1e-7``), which is the gradient norm at which the training terminates and  
```round_result``` (default: ``True``), which, if set to True, rounds the result to two decimal places  
  
The resulting MHN is returned by the ```train()``` method, but can also be obtained
from the ```result``` parameter:
```python
new_mhn = opt.result
```

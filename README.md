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

## A quick overview

The package contains the original MHN functions implemented in Python. You import them from ``mhn.original``:

```python
from mhn.original import Likelihood, ModelConstruction, RegularizedOptimization, UtilityFunctions
```
It contains also functions to compute the Fisher information for a given MHN and use
Natural Gradient Descent to train a new model. You can use those functions by importing
```python
from mhn.original import FisherFunctions
```
Lastly, you can train a MHN using state-space restriction. The corresponding functions
can be imported with
```python
from mhn.ssr import StateSpaceRestrictionCython, state_storage
```

"""
This part of the mhn package contains functions to compute and work with MHNs on the full state-space.

The structure of this part of the package is very similar to the original R implementation by Schill et al. (2019):

Likelihood:
    contains functions to compute the log-likelihood score and its gradient without state-space restriction as
    well as functions for matrix-vector multiplications with the transition rate matrix and [I-Q]^(-1)

ModelConstruction:
    contains functions to generate random MHNs, build their transition rate matrix Q, the diagonal of Q, and to generate
    and independence model for a given distribution

RegularizedOptimization:
    contains functions to learn an cMHN for a given data distribution, implements the L1 regularization

UtilityFunctions:
    contains functions useful for preprocessing data to be used as training data

PerformanceCriticalCode:
    contains functions that dominate the runtime of score and gradient computations and must therefore be implemented
    efficiently
"""
# author(s): Stefan Vocht


from . import PerformanceCriticalCode
from . import Likelihood
from . import ModelConstruction
from . import RegularizedOptimization
from . import UtilityFunctions
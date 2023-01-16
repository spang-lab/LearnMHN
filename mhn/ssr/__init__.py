"""
This part of the package contains functions that use state-space restriction (SSR) to become more efficient and
compute scores and their gradients that are needed for learning MHNs much faster.
"""

from . import state_storage
from . import state_space_restriction
from . import matrix_exponential

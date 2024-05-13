# by Stefan Vocht
#
# this file acts like a C header file for ModelConstruction.pyx
#
cimport numpy as np
np.import_array()

cpdef np.ndarray[np.double_t, ndim=1] q_subdiag(double[:, :] theta, int i)


cpdef np.ndarray[np.double_t, ndim=1] q_diag(double[:, :] theta)
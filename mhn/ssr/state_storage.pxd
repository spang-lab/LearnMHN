# by Stefan Vocht
#
# this .pxd file is the cython header file for state_storage.pyx
#


# STATE_SIZE is defined in setup.py, 
# the maximum number n of genes the MHN can handle is 32 * STATE_SIZE
cdef extern from *:
    """
    typedef struct {
        uint32_t parts[STATE_SIZE];
    } State;
    """
    ctypedef struct State:
        unsigned int parts[STATE_SIZE]


cdef class StateStorage:
    cdef State *states
    cdef int data_size
    cdef int gene_num
    cdef int max_mutation_num


cdef class StateAgeStorage(StateStorage):
    cdef double *state_ages





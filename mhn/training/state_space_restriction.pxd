# by Stefan Vocht
#
# this is the header file for state_space_restriction.pyx

from mhn.ssr.state_containers cimport State

cdef int get_mutation_num(State *state)

cdef void restricted_q_vec(double[:, :] theta, double[:] x, State *state, double *yout, bint diag= *, bint transp = *)

cdef void restricted_q_diag(double[:, :] theta, State *state, double *dg)
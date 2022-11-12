# by Stefan Vocht
#
# this file acts like a C header file for ModelConstruction.pyx
#


cpdef q_subdiag(double[:, :] theta, int i):
    """
    Creates a single subdiagonal of Q from the ith row in Theta

    :return: subdiagonal of Q corresponding to the ith row of Theta
    """


cpdef q_diag(double[:, :] theta):
    """
    get the diagonal of Q

    :param theta: theta representing the MHN
    """
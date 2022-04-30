/*
by Stefan Vocht

this file contains C functions that can be called by functions in Likelihood.py to increase performance

The functions are modified versions of the code one can find in InlineFunctions.R
*/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>


// extern BLAS functions used for speed up
extern void dcopy_(int* n, double* dx, int* incx, double* dy, int* incy);
extern void dscal_(int* n, double* da, double* dx, int* incx);
extern double ddot_(int* n, double* dx, int* incx, double* dy, int* incy);
extern void daxpy_(int *n, double *da, double *dx, int *incx, double *dy, int *incy);


/*
function c_kronvec:

Used to calulate Q (as a kronecker product) multiplied with a vector x

Arguments:
ptheta: pointer to the Theta-Matrix
i: the row of Theta which is used for the product
px: array to multiply with
diag: set true, if the diagonal should be considered for the product
transp: set true, if you need the product with the transposed matrix
n: number of rows/columns of Theta
pout: allocated memory of size 2^n, where the result is stored
*/
void c_kronvec(double *ptheta, int i, double *px, bool diag, bool transp, int n, double *pout){

  ptheta = ptheta + i * n; // get the pointer to the ith row of theta

  int nx      = 1 << n;  // fast way to calculate 2^n
  int nxhalf  = nx/2;
  double mOne = -1;
  double zero = 0;

  double *ptmp = malloc(nx*sizeof(double));
  double *px1,*px2;

  // instead of copying the shuffled vector back into the original
  // we change the following pointers in such a way that we do not have to copy the data back
  double *shuffled_vec, *old_vec, *swap_vec;

  double theta;
  int incx     = 1;
  int incx2    = 2;
  int j;

  // this condition is needed for the final result to be in the correct array
  if (n % 2 == 1){
    old_vec = ptmp;
    shuffled_vec = pout;
  } else {
    old_vec = pout;
    shuffled_vec = ptmp;
  }


  // the argument of the function must not be modified
  dcopy_(&nx,px,&incx,old_vec,&incx);

  for (j=0; j<n; j++) {

    // matrix shuffle

    dcopy_(&nxhalf,old_vec,&incx2,shuffled_vec,&incx);
    dcopy_(&nxhalf,old_vec+1,&incx2,shuffled_vec+nxhalf,&incx);

    theta = exp(ptheta[j]);
    px1   = shuffled_vec;
    px2   = shuffled_vec + nxhalf;

    // Kronecker product (daxpby is slightly slower than dcopy_ + dscal_)

    if (j == i) {
      if (!transp) {
        dcopy_(&nxhalf,px1,&incx,px2,&incx);
        dscal_(&nxhalf,&theta,px2,&incx);
        if (diag) {
          dcopy_(&nxhalf,px2,&incx,px1,&incx);
          dscal_(&nxhalf,&mOne,px1,&incx);
        } else {
          dscal_(&nxhalf,&zero,px1,&incx);
        }
      } else {
        if (diag) {
          theta *= -1;
          daxpy_(&nxhalf, &mOne, px2, &incx, px1, &incx);
          dscal_(&nxhalf, &theta, px1, &incx);
          dscal_(&nxhalf, &zero, px2, &incx);
        } else {
          dcopy_(&nxhalf,px2,&incx,px1,&incx);
          dscal_(&nxhalf,&theta,px1,&incx);
          dscal_(&nxhalf,&zero,px2,&incx);
        }
      }
    } else {
      dscal_(&nxhalf,&theta,px2,&incx);
    }  

    swap_vec = old_vec;
    old_vec = shuffled_vec;
    shuffled_vec = swap_vec;
  }

  free(ptmp);
}


void c_loop_j(int i, int n, double* pr, double *pG){
  int nx       = 1 << n; //fast way to calculate 2^n
  int nxhalf   = nx/2;

  pG = pG + i*n;  // get ith row of pG

  double one   = 1.;
  double *ptmp = malloc(nx*sizeof(double));

  // instead of copying the shuffled vector back into the original
  // we change the following pointers in such a way that we do not have to copy the data back
  double *shuffled_vec, *old_vec, *swap_vec;

  int incx     = 1;
  int incx2    = 2;
  int inc0     = 0;
  int j;

  old_vec = pr;
  shuffled_vec = ptmp;

  for (j=0; j<n; j++) {
    // matrix shuffle
    dcopy_(&nxhalf,old_vec,&incx2,shuffled_vec,&incx);
    dcopy_(&nxhalf,old_vec+1,&incx2,shuffled_vec+nxhalf,&incx);

    // sums as dot products with 1
    pG[j] = ddot_(&nxhalf,shuffled_vec+nxhalf,&incx,&one,&inc0);
    if (j == i) pG[j] += ddot_(&nxhalf,shuffled_vec,&incx,&one,&inc0);

    swap_vec = old_vec;
    old_vec = shuffled_vec;
    shuffled_vec = swap_vec;
  }

  free(ptmp);
}
#include <stdio.h>

#include "jacobi.h"

#if __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

__global__ void jacobistep(double *psinew, double *psi, int m, int n) {
  int i, j;

  for (i = 1; i <= m; i++) {
    for (j = 1; j <= n; j++) {
      psinew[i * (m + 2) + j] = 0.25 * (psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] + psi[i * (m + 2) + j - 1] + psi[i * (m + 2) + j + 1]);
    }
  }
}

__global__ void deltasq(double *dsq, double *newarr, double *oldarr, int m, int n) {
  int i, j;

  double tmp;

  for (i = 1; i <= m; i++) {
    for (j = 1; j <= n; j++) {
      tmp = newarr[i * (m + 2) + j] - oldarr[i * (m + 2) + j];
      atomicAddDouble(dsq, tmp * tmp);
    }
  }
}

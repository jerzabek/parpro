#include <math.h>
#include <stdio.h>

#include <cmath>

#include "jacobi.h"

// Dokumentacija preporuča ovaj snippet za atomicAdd koji podržava double
#if __CUDA_ARCH__ < 600
__device__ double doubleAtomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull =
      (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

__global__ void jacobistep(double *psinew, double *psi, int m, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (i <= m && j <= n) {
    psinew[i * (m + 2) + j] = 0.25 * (psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] + psi[i * (m + 2) + j - 1] + psi[i * (m + 2) + j + 1]);
  }
}

__global__ void deltasq(double *dsq, double *newarr, double *oldarr, int m, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (i <= m && j <= n) {
    double tmp = newarr[i * (m + 2) + j] - oldarr[i * (m + 2) + j];

    // double epsilon = 1e-10;
    // if (fabs(tmp) > epsilon) {
    //   printf("i: %d, j: %d tmp: %f ", i, j, tmp);
    // }
    // if (fabs(newarr[i * (m + 2) + j]) > epsilon) {
    //   printf("new arr: %f ", newarr[i * (m + 2) + j]);
    // }
    // if (fabs(oldarr[i * (m + 2) + j]) > epsilon) {
    //   printf("old arr: %f \n", oldarr[i * (m + 2) + j]);
    // }

    doubleAtomicAdd(dsq, tmp * tmp);
  }
}

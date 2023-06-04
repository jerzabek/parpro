#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "arraymalloc.h"
#include "boundary.h"
#include "cfdio.h"
#include "jacobi.h"

__global__ void copy_psitmp_to_psi_device(double *d_psitmp, double *d_psi, int m, int n) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i <= m && j <= n) {
    d_psi[i * (m + 2) + j] = d_psitmp[i * (m + 2) + j];
  }
}

int main(int argc, char **argv) {
  int printfreq = 1000;  // output frequency
  double error, bnorm;
  double tolerance = 0.0;  // tolerance for convergence. <=0 means do not check

  // main arrays
  double *psi;
  // temporary versions of main arrays
  double *psitmp;

  // command line arguments
  int scalefactor, numiter;

  // simulation sizes
  int bbase = 10;
  int hbase = 15;
  int wbase = 5;
  int mbase = 32;
  int nbase = 32;

  int irrotational = 1, checkerr = 0;

  int m, n, b, h, w;
  int iter;
  int i, j;

  double tstart, tstop, ttot, titer;

  // do we stop because of tolerance?
  if (tolerance > 0) {
    checkerr = 1;
  }

  // check command line parameters and parse them

  if (argc < 3 || argc > 4) {
    printf("Usage: cfd <scale> <numiter>\n");
    return 0;
  }

  scalefactor = atoi(argv[1]);
  numiter = atoi(argv[2]);

  if (!checkerr) {
    printf("Scale Factor = %i, iterations = %i\n", scalefactor, numiter);
  } else {
    printf("Scale Factor = %i, iterations = %i, tolerance= %g\n", scalefactor, numiter, tolerance);
  }

  printf("Irrotational flow\n");

  // Calculate b, h & w and m & n
  b = bbase * scalefactor;
  h = hbase * scalefactor;
  w = wbase * scalefactor;
  m = mbase * scalefactor;
  n = nbase * scalefactor;

  printf("Running CFD on %d x %d grid in serial\n", m, n);

  // allocate arrays
  psi = (double *)malloc((m + 2) * (n + 2) * sizeof(double));
  psitmp = (double *)malloc((m + 2) * (n + 2) * sizeof(double));

  double *d_psi, *d_psitmp, *d_dsq;
  double h_dsq = 0.0;

  // zero the psi array
  for (i = 0; i < m + 2; i++) {
    for (j = 0; j < n + 2; j++) {
      psi[i * (m + 2) + j] = 0.0;
    }
  }
  for (i = 0; i < m + 2; i++) {
    for (j = 0; j < n + 2; j++) {
      psitmp[i * (m + 2) + j] = 0.0;
    }
  }

  cudaMalloc(&d_psi, (m + 2) * (n + 2) * sizeof(double));
  cudaMalloc(&d_psitmp, (m + 2) * (n + 2) * sizeof(double));
  cudaMalloc(&d_dsq, sizeof(double));

  cudaMemcpy(d_psi, psi, (m + 2) * (n + 2) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_psitmp, psitmp, (m + 2) * (n + 2) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dsq, &h_dsq, sizeof(double), cudaMemcpyHostToDevice);

  // set the psi boundary conditions
  boundarypsi(psi, m, n, b, h, w);

  // compute normalisation factor for error
  bnorm = 0.0;

  for (i = 0; i < m + 2; i++) {
    for (j = 0; j < n + 2; j++) {
      bnorm += psi[i * (m + 2) + j] * psi[i * (m + 2) + j];
    }
  }
  bnorm = sqrt(bnorm);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // begin iterative Jacobi loop
  printf("\nStarting main loop...\n\n");
  tstart = gettime();

  for (iter = 1; iter <= numiter; iter++) {
    // calculate psi for next iteration
    jacobistep<<<numBlocks, threadsPerBlock>>>(d_psitmp, d_psi, m, n);
    cudaDeviceSynchronize();
    // calculate current error if required
    if (checkerr || iter == numiter) {
      deltasq<<<numBlocks, threadsPerBlock>>>(d_dsq, d_psitmp, d_psi, m, n);

      printf("finish deltasq ");
      cudaMemcpy(&h_dsq, d_dsq, sizeof(double), cudaMemcpyDeviceToHost);
      printf("h_dsq = %g ", h_dsq);
      error = sqrt(h_dsq);
      error = error / bnorm;
      printf("Iteration %d, error = %g\n", iter, error);
    }

    if (checkerr) {
      if (error < tolerance) {
        printf("Converged on iteration %d ", iter);
        break;
      }
    }

    // cudaMemcpy(psitmp, d_psitmp, (m + 2) * (n + 2) * sizeof(double), cudaMemcpyDeviceToHost);

    copy_psitmp_to_psi_device<<<numBlocks, threadsPerBlock>>>(d_psitmp, d_psi, m, n);

    // cudaMemcpy(d_psi, psi, (m + 2) * (n + 2) * sizeof(double), cudaMemcpyHostToDevice);

    // printf("Iteration %d, error = %g ", iter, error);

    if (iter % printfreq == 0) {
      if (!checkerr) {
        printf("Completed iteration %d\n", iter);
      } else {
        printf("Completed iteration %d, error = %g\n", iter, error);
      }
    }

  }  // iter

  if (iter > numiter) iter = numiter;

  tstop = gettime();

  ttot = tstop - tstart;
  titer = ttot / (double)iter;

  // print out some stats
  printf("\n... finished\n");
  printf("After %d iterations, the error is %g\n", iter, error);
  printf("Time for %d iterations was %g seconds\n", iter, ttot);
  printf("Each iteration took %g seconds\n", titer);

  // output results
  // writedatafiles(psi,m,n, scalefactor);
  // writeplotfile(m,n,scalefactor);

  cudaFree(d_psi);
  cudaFree(d_psitmp);
  cudaFree(d_dsq);

  // free un-needed arrays
  free(psi);
  free(psitmp);
  printf("... finished\n");

  return 0;
}

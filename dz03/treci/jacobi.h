//#include <nvtx3/nvToolsExt.h>

__global__ void jacobistep(double *psinew, double *psi, int m, int n);

__global__ void deltasq(double *dsq, double *newarr, double *oldarr, int m, int n);

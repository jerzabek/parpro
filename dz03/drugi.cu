#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256

/**
 * Ubrzanje slijednog algoritma u usporedbi sa paralelnim algoritmom

  N	          SEQ	      Parallel	Ubrzanje
  10000	      1,9	      0,4	      4,75
  100000	    17,7	    1,6	      11,0625
  1000000	    175,8	    13,2	    13,31818182
  10000000	  1357,4	  127,7	    10,62960063

*/

__global__ void setup_kernel(curandState *state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  curand_init(1337, i, i, &state[i]);
}

__global__ void generate_kernel(curandState *state, unsigned int *counters) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  float x = curand_uniform(&state[i]);
  float y = curand_uniform(&state[i]);

  if (x * x + y * y <= 1.0f) atomicAdd(counters, 1);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Missing argument: %s <number of points>\n", argv[0]);
    return 1;
  }

  int N = atoi(argv[1]);

  curandState *devStates;
  unsigned int *devCounters;
  unsigned int hostCounters = 0;

  cudaEvent_t start, stop;
  float time;

  cudaMalloc((void **)&devCounters, sizeof(unsigned int));
  cudaMemset(devCounters, 0, sizeof(unsigned int));

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  cudaMalloc((void **)&devStates, THREADS_PER_BLOCK * sizeof(curandState));

  setup_kernel<<<1, THREADS_PER_BLOCK>>>(devStates);

  for (int i = 0; i < N; i += THREADS_PER_BLOCK) {
    generate_kernel<<<1, THREADS_PER_BLOCK>>>(devStates, devCounters);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(&hostCounters, devCounters, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  printf("Pi is basically = %f\n", 4.0 * hostCounters / N);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Time:  %3.1f ms \n", time);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(devStates);
  cudaFree(devCounters);

  return 0;
}

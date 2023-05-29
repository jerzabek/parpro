#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100000
#define console printf

__global__ void calcDistance(int num_cities, float *x, float *y, float *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < num_cities) {
    float sum = 0;

    for (int j = 0; j < num_cities; j++) {
      float dx = x[i] - x[j];
      float dy = y[i] - y[j];
      sum += sqrt(dx * dx + dy * dy);
    }

    output[i] = sum / (num_cities - 1);
  }
}

int main() {
  srand(time(NULL));

  float *x = (float *)malloc(sizeof(float) * N);
  float *y = (float *)malloc(sizeof(float) * N);
  float *output = (float *)malloc(sizeof(float) * N);

  for (int i = 0; i < N; i++) {
    x[i] = rand() / (float)RAND_MAX;
    y[i] = rand() / (float)RAND_MAX;
  }

  float *device_x, *device_y, *device_output;

  cudaMalloc(&device_x, sizeof(float) * N);
  cudaMalloc(&device_y, sizeof(float) * N);
  cudaMalloc(&device_output, sizeof(float) * N);

  cudaMemcpy(device_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(device_y, y, sizeof(float) * N, cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = ceil(N / blockSize);
  calcDistance<<<numBlocks, blockSize>>>(N, device_x, device_y, device_output);

  cudaMemcpy(output, device_output, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // Compute average distance
  float avg = 0.0f;
  for (int i = 0; i < N; i++) {
    avg += output[i];
  }
  avg /= N;

  printf("Prosjecna udaljenost je %f\n", avg);

  free(x);
  free(y);
  free(output);

  cudaFree(device_x);
  cudaFree(device_y);
  cudaFree(device_output);

  return 0;
}

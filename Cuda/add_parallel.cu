#include <stdio.h>
#include <unistd.h>

#define ARRSIZ 100
#define INIT_TID (threadIdx.x + blockIdx.x * blockDim.x)
#define ITER_TID (blockDim.x * gridDim.x)

__global__ void add_arrays(int *c, int *a, int *b, int n) {
  int tID = INIT_TID;
  while (tID < n) {
    c[tID] = a[tID] + b[tID];
    tID += ITER_TID;
  }
}

__global__ void print_array(int *c, int n) {
  int tID = INIT_TID;
  while (tID < n) {
    printf("%5d", c[tID]);
    tID += ITER_TID;
  }
}

int main() {
  const int n = ARRSIZ * 10;
  int a[ARRSIZ], b[ARRSIZ];
  int *dev_a, *dev_b, *dev_c;

  cudaMalloc(&dev_a, ARRSIZ * sizeof(int));
  cudaMalloc(&dev_b, ARRSIZ * sizeof(int));
  cudaMalloc(&dev_c, ARRSIZ * sizeof(int));

  for (int i = 0; i < n; i += ARRSIZ) {
    for (int j = 0; j < ARRSIZ; ++j) {
      a[j] = (i + j) * 2;
      b[j] = (i + j) * 3;
    }

    cudaMemcpy(dev_a, a, ARRSIZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, ARRSIZ * sizeof(int), cudaMemcpyHostToDevice);

    add_arrays<<<8, 16>>>(dev_c, dev_a, dev_b, ARRSIZ);
    print_array<<<8, 16>>>(dev_c, ARRSIZ);
  }

  sleep(1);
  printf("\n");

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}

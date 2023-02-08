#include <stdint.h>
#include <stdio.h>

#define MODULE_N 2
#define CUDA_BLOCK_N 65535

typedef uint8_t uint1_t;

__global__ void rand_uintN(uint8_t *r) { *r = blockIdx.x % MODULE_N; }

void rand_uint1s(uint1_t *gamma, int n);
void print_uint1s(uint1_t *gamma, int n);
void print_uint1s_count(uint1_t *gamma, int n);

int main() {
  const int n = 1024;
  uint1_t gamma[n];

  rand_uint1s(gamma, n);

  print_uint1s(gamma, n);
  print_uint1s_count(gamma, n);
  
  return 0;
}

void rand_uint1s(uint1_t *gamma, int n) {
  const int num_count = n * MODULE_N;

  uint8_t raw_rand[num_count];
  uint8_t *dev_r;

  memset(raw_rand, 0, sizeof(raw_rand));
  cudaMalloc(&dev_r, sizeof(uint8_t));
  for (int i = 0; i < num_count; i++) {
    rand_uintN<<<CUDA_BLOCK_N, 1>>>(dev_r);
    cudaMemcpy(raw_rand + i, dev_r, sizeof(uint8_t), cudaMemcpyDeviceToHost);
  }
  cudaFree(dev_r);

  for (int i = 0; i < num_count; i += MODULE_N) {
    int sum = 0;
    for (int j = 0; j < MODULE_N; ++j) {
      sum += raw_rand[i + j];
    }
    gamma[i / MODULE_N] = sum % 2;
  }
}

void print_uint1s_count(uint1_t *gamma, int n) {
  int count[2];
  memset(count, 0, sizeof(count));

  for (int i = 0; i < n; ++i) {
    count[gamma[i]]++;
  }

  for (int i = 0; i < 2; ++i) {
    printf("[%d] = %d\n", i, count[i]);
  }
}

void print_uint1s(uint1_t *gamma, int n) {
  for (int i = 0; i < n; ++i) {
    printf("%d", gamma[i]);
  }
  printf("\n");
}

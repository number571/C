#include <stdio.h>

__global__ void mul_arrays(int *c, int *a, int *b, int n) {
    int tID = blockIdx.x;
    if (tID < n) {
        c[tID] = a[tID] * b[tID];
    }
}

int main() {
    const int n = 10;
    int a[n], b[n], c[n];
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc(&dev_a, n*sizeof(int));
    cudaMalloc(&dev_b, n*sizeof(int));
    cudaMalloc(&dev_c, n*sizeof(int));

    for (int i = 0; i < n; ++i) {
        a[i] = i*2;
        b[i] = i*3;
    }

    cudaMemcpy(dev_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n*sizeof(int), cudaMemcpyHostToDevice);

    mul_arrays<<<n,1>>>(dev_c, dev_a, dev_b, n);

    cudaMemcpy(c, dev_c, n*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        printf("%d ", c[i]);
    }
    printf("\n");

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

#include <stdio.h>

__global__ void cuda_mul(int *c, int a, int b){
    *c = a * b;
}

int main() {
    int *dev_c, c;

    cudaMalloc((void**)&dev_c, sizeof(int));
    cuda_mul<<<1,1>>>(dev_c, 5, 6); 
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);

    printf("a * b = %d\n", c);
    return 0;
}

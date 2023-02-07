#include <stdio.h>

int main() {
    int count;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&count);
    

    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("%s, %luMib\n", prop.name, prop.totalGlobalMem/(1<<20));
    }

    return 0;
}

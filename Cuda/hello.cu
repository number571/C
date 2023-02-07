#include <stdio.h>
#include <unistd.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
    sleep(1);
    return 0;
}

#include <stdio.h>

void swap(int *x, int *y);

int main(void) {

    int x = 5, y = 10;
    printf("%d %d\n", x,y);

    swap(&x, &y);
    printf("%d %d\n", x,y);

    return 0;
}

void swap(int *x, int *y) {
    *x ^= *y ^= *x ^= *y;
}

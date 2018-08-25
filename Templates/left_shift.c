#include <stdio.h>
#define N 10U

void left_shift (int arr[], unsigned len);
void print_array (int arr[], unsigned len);

int main (void) {
    int array[N] = {1,2,3,4,5,6,7,8,9,10};
    unsigned i;
    for (i = 0; i < 3; i++)
        left_shift(array, N);
    print_array(array, N);
    return 0;
}

void left_shift (int arr[], unsigned len) {
    int temp = arr[0];
    unsigned i;
    for (i = 1; i < len; i++)
        arr[i-1] = arr[i];
    arr[i-1] = temp;
}

void print_array (int arr[], unsigned len) {
    unsigned i;
    for (i = 0; i < len; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

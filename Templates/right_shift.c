#include <stdio.h>
#define N 10U

void right_shift (int arr[], unsigned len);
void print_array (int arr[], unsigned len);

int main (void) {
    int array[N] = {1,2,3,4,5,6,7,8,9,10};
    unsigned i;
    for (i = 0; i < 3; i++)
        right_shift(array, N);
    print_array(array, N);
    return 0;
}

void right_shift (int arr[], unsigned len) {
    int temp = arr[len-1];
    unsigned i;
    for (i = len-1; i > 0; i--)
        arr[i] = arr[i-1];
    arr[0] = temp;
}

void print_array (int arr[], unsigned len) {
    unsigned i;
    for (i = 0; i < len; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

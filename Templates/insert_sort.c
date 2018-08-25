#include <stdio.h>
#define N 10U

void insert_sort (int arr[], unsigned len);
void print_array (int arr[], unsigned len);

int main (void) {
    int array[N] = {5,7,3,4,1,2,9,0,6,8};
    insert_sort(array, N);
    print_array(array, N);
    return 0;
}

void insert_sort (int arr[], unsigned len) {
    int temp;
    unsigned i, j;
    for (i = 1; i < len; i++)
        for (j = i; j > 0 && arr[j] < arr[j-1]; j--) {
            temp = arr[j];
            arr[j] = arr[j-1];
            arr[j-1] = temp;
        }
}

void print_array (int arr[], unsigned len) {
    unsigned i;
    for (i = 0; i < len; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

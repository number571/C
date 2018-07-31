#include <stdio.h>
#define SIZE 10

void qsort_(int v[], int left, int right);

int main (void) {
    int index;
    int arr[SIZE] = {5,6,4,7,3,8,9,0,2,1};
    qsort_(arr, 0, SIZE-1);
    for (index = 0; index < SIZE; index++)
        printf("%d ", arr[index]);
    printf("\n");
    return 0;
}

void qsort_(int v[], int left, int right) {
    int i, last;
    void swap(int v[], int i, int j);
    if (left >= right) return;
    swap(v, left, (left + right) / 2);
    last = left;
    for (i = left+1; i <= right; i++)
        if (v[i] < v[left])
            swap(v, ++last, i);
    swap(v, left, last);
    qsort_(v, left, last-1);
    qsort_(v, last+1, right);
}

void swap(int v[], int i, int j) {
    int temp;
    temp = v[i], v[i] = v[j], v[j] = temp;
}

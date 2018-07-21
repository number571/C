#include <stdio.h>

void shellsort(int v[], int n);

int main (void) {
    int array[10] = {5,2,4,1,3,8,9,0,7,6};
    shellsort(array, 10);
    
    int index;
    for (index = 0; index < 10; index++)
        printf("%d ", array[index]);
    printf("\n");

    return 0;
}

void shellsort(int v[], int n) {
    int gap, i, j, temp;
    for (gap = n/2; gap > 0; gap /= 2)
        for (i = gap; i < n; i++)
            for (j = i-gap; j >= 0 && v[j] > v[j+gap]; j -= gap) {
                temp = v[j];
                v[j] = v[j+gap];
                v[j+gap] = temp;
            }
}

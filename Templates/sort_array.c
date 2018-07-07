#include <stdio.h>
#define SIZE 10

void sort   (int array[]);
void reverse(int array[]);
void output (int array[]);

int main (void) {
    int arr[SIZE] = {4,3,8,5,1,0,9,6,7,2};
    sort(arr); reverse(arr); output(arr);
    return 0;
}

void sort (int array[]) {
    int temp;
    for (unsigned index = 1; index < SIZE; index ++)
        for (unsigned twindex = 1; twindex < SIZE; twindex ++)
            if (array[twindex] < array[twindex - 1]) {
                temp = array[twindex];
                array[twindex] = array[twindex - 1];
                array[twindex - 1] = temp;
            }
}

void reverse (int array[]) {
    int temp;
    for (unsigned index = 0; index < SIZE/2; index ++) {
        temp = array[index];
        array[index] = array[SIZE - index - 1];
        array[SIZE - index - 1] = temp;
    }
}

void output (int array[]) {
    for (unsigned index = 0; index < SIZE; index ++)
        printf("%hd ", array[index]);
    printf("\n");
}

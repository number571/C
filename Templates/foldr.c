#include <stdio.h>
#define SIZE 6

double foldr (
    double (*func) (double, double), 
    double num,
    double array[],
    unsigned int length
);
double mul (double x, double y);

int main (void) {
    unsigned int i;

    double array[SIZE] = {1,2,3,4,5,6};
    printf("%.0lf\n", foldr(mul, 1, array, SIZE));

    return 0;
}

double foldr (
    double (*func) (double, double), 
    double num,
    double array[],
    unsigned int length
) {
    double *p = array;
    while (p < array + length)
        num = func(num, *p++);
    return num;
}

double mul (double x, double y) {
    return x * y;
}

#include <stdio.h>
#define SIZE 10

void map (
    double (*func) (double, double), 
    double value, 
    double array[], 
    unsigned int length
);
double pow_ (double num, double p);

int main (void) {
    unsigned int i;

    double array[SIZE] = {1,2,3,4,5,6,7,8,9,10};
    map(pow_, 2, array, SIZE);

    for (i = 0; i < SIZE; i++)
        printf("%.0lf ", array[i]);
    printf("\n");

    return 0;
}

void map (
    double (*func) (double, double), // Функция
    double value, // Значение в функцию
    double array[], // Массив
    unsigned int length // Длина массива
) {
    double *p = array;
    while (p < array + length)
        *p++ = func(*p, value);
}

double pow_ (double num, double p) {
    p = (int) p;
    if (p > 0)
        while (p-- > 1)
            num *= num;
    else if (p < 0) {
        num = 1 / num;
        while (p++ < -1)
            num *= num;
    }
    else return 1;
    return num;
}

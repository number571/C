#include <stdio.h>

#define STR "abc"

#define LEN 3
#define SIZE 3

void variants (char *str, unsigned len, unsigned size);
double power (double num, int p);

int main (void) {
    variants(STR, LEN, SIZE);
    return 0;
}

void variants (char *str, unsigned len, unsigned size) {
    int i, j;
    for (i = 0; i < power(size, len); ++i) {
        for (j = len-1; j >= 0; --j)
           putchar(str[(int)(i/(power(size,j))) % size]);
        putchar('\n');
    }
}

double power (double num, int p) {
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

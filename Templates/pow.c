#include <stdio.h>

double pow_ (double num, int p);

int main (void) {
    printf("%lf\n", pow_(5, 2));
    return 0;
}

double pow_ (double num, int p) {
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

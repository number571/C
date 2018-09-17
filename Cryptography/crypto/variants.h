#include <stdio.h>
#include <string.h>
#include <math.h>

void variants (char *str, unsigned char size) {
    unsigned char length = strlen(str);
    unsigned int i;
    int j;
    for (i = 0; i < pow(length, size); ++i) {
        for (j = size - 1; j >= 0; --j)
           putchar(str[(char) (i / (pow(length, j)) ) % length]);
        putchar('\n');
    }
}

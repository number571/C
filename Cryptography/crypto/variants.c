#include <stdio.h>
#include <string.h>
#include <math.h>

extern void variants (const char * const str, const unsigned char size) {
    const unsigned char length = strlen(str);
    int x, y;
    for (x = 0; x < pow(length, size); ++x) {
        for (y = size - 1; y >= 0; --y)
           putchar(str[(char) (x / (pow(length, y)) ) % length]);
        putchar('\n');
    }
}
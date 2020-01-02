#include <stdio.h>
#include <stdint.h>

int8_t L(uint64_t a, uint64_t p);
uint64_t addmod(uint64_t x, uint64_t y, uint64_t n);
uint64_t mulmod(uint64_t x, uint64_t y, uint64_t n);
uint64_t powmod(uint64_t x, uint64_t y, uint64_t n);
void print_group(uint64_t n, uint64_t (*groupf) (uint64_t, uint64_t, uint64_t));

int main(void) {
    uint64_t p = 11;
    print_group(p, powmod);
    return 0;
}

/*
 1 = Квадратичный вычет;
-1 = Квадратичный невычет;
 0 = a | p;
*/
int8_t L(uint64_t a, uint64_t p) {
    uint64_t x = powmod(a, (p - 1) / 2, p);
    return (x == p - 1) ? -1 : x;
}

uint64_t addmod(uint64_t x, uint64_t y, uint64_t n) {
    return (x + y) % n;
}

uint64_t mulmod(uint64_t x, uint64_t y, uint64_t n) {
    return (x * y) % n;
}

uint64_t powmod(uint64_t x, uint64_t y, uint64_t n) {
    uint64_t r = 1;
    while (y) {
        if (y & 0x01) {
            r = (r * x) % n;
        }
        x = (x * x) % n;
        y >>= 1;
    }
    return r;
}

void print_group(uint64_t n, uint64_t (*groupf) (uint64_t, uint64_t, uint64_t)) {
    printf(" Group: %d\n", n);
    printf(" Z |");
    for (uint64_t i = 0; i < n; ++i) {
        printf("%2d |", i);
    }
    putchar('\n');
    for (uint64_t i = 1; i < n; ++i) {
        printf("%2d |", i);
        for (uint64_t j = 0; j < n; ++j) {
            printf("%2d |", (*groupf)(i, j, n));
        }
        putchar('\n');
    }
}

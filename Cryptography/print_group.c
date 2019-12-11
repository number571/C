#include <stdio.h>
#include <stdint.h>

uint64_t addmod(uint64_t x, uint64_t y, uint64_t n);
uint64_t mulmod(uint64_t x, uint64_t y, uint64_t n);
uint64_t powmod(uint64_t x, uint64_t y, uint64_t n);
void print_group(uint8_t n, uint64_t (*groupf) (uint64_t, uint64_t, uint64_t));

int main(void) {
    print_group(11, powmod);
    return 0;
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

void print_group(uint8_t n, uint64_t (*groupf) (uint64_t, uint64_t, uint64_t)) {
    printf(" Group: %d\n", n);
    printf(" - |");
    for (uint8_t i = 0; i < n; ++i) {
        printf("%2d |", i);
    }
    putchar('\n');
    for (uint8_t i = 1; i < n; ++i) {
        printf("%2d |", i);
        for (uint8_t j = 0; j < n; ++j) {
            printf("%2d |", (*groupf)(i, j, n));
        }
        putchar('\n');
    }
}

#include <stdio.h>
#include <stdint.h>

uint64_t inv_number(uint64_t a, uint64_t b);

int main(void) {
    uint64_t x = 7;
    uint64_t y = 8;
    uint64_t z = inv_number(x, y);
    printf("%lu * %lu = %lu (mod %lu)\n", x, z, x * z % y, y);
    return 0;
}

// extended euclid algorithm
// x = a^(-1) mod b
uint64_t inv_number(uint64_t a, uint64_t b) {
    uint64_t tx = 0, x0 = 1, x1 = 0;
    uint64_t q = 0, r = 0;
    uint64_t tb = b;

    while (b != 0) {
        q = a / b, r = a % b;
        tx = x0 - q * x1;
        a = b, b = r;
        x0 = x1, x1 = tx;
    }

    return (x0 + tb) % tb;
}

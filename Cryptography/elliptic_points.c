#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define MODULE 17

int main(void) {
    // elliptic curve = E(2,2)17
    for (uint64_t y = 0; y < MODULE; ++y) {
        for (uint64_t x = 0; x < MODULE; ++x) {
            uint64_t left  = (uint64_t)pow((double)y, 2) % MODULE;
            uint64_t right = ((uint64_t)pow((double)x, 3) + 2*x + 2) % MODULE;
            if (left == right) {
                printf("(%d, %d)\n", x, y);
            }
        }
    }
    return 0;
}

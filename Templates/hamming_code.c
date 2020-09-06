#include <stdio.h>
#include <stdint.h>

static uint8_t hamming_encode(uint8_t x);
static uint8_t hamming_decode(uint8_t x);

static void set_bit(uint8_t *x, _Bool bit, size_t i);
static _Bool get_bit(uint8_t x, size_t i);

static void print_bits(uint8_t x);

int main(void) {
    uint8_t num = 15;

    printf("Message:\t");
    print_bits(num);

    uint8_t hcode = hamming_encode(num);
    printf("Encoded:\t");
    print_bits(hcode);

    set_bit(&hcode, 0, 3);
    printf("Noise:\t\t");
    print_bits(hcode);

    uint8_t dec = hamming_decode(hcode);
    printf("Decoded:\t");
    print_bits(dec);

    return 0;
}

// input:
// |  1  2  3  4  5  6  7  8
// | p1 p2 d1 p3 d2 d3 d4 n0
// output: 
// |  1  2  3  4
// | d4 d3 d2 d1
static uint8_t hamming_decode(uint8_t x) {
    uint8_t i = 0;
    uint8_t y = 0;

    set_bit(&i, get_bit(x, 1) ^ get_bit(x, 3) ^ get_bit(x, 5) ^ get_bit(x, 7), 1);
    set_bit(&i, get_bit(x, 2) ^ get_bit(x, 3) ^ get_bit(x, 6) ^ get_bit(x, 7), 2);
    set_bit(&i, get_bit(x, 4) ^ get_bit(x, 5) ^ get_bit(x, 6) ^ get_bit(x, 7), 3);

    if (i) {
        set_bit(&x, !get_bit(x, i), i);
    }

    set_bit(&y, get_bit(x, 3), 4);
    set_bit(&y, get_bit(x, 5), 3);
    set_bit(&y, get_bit(x, 6), 2);
    set_bit(&y, get_bit(x, 7), 1);

    return y;
}

// input: 
// |  1  2  3  4
// | d4 d3 d2 d1
// output:
// |  1  2  3  4  5  6  7  8
// | p1 p2 d1 p3 d2 d3 d4 n0
static uint8_t hamming_encode(uint8_t x) {
    uint8_t y = 0;
    if (x >= 16) {
        return y;
    }

    set_bit(&y, get_bit(x, 4), 3);
    set_bit(&y, get_bit(x, 3), 5);
    set_bit(&y, get_bit(x, 2), 6);
    set_bit(&y, get_bit(x, 1), 7);

    set_bit(&y, get_bit(y, 3) ^ get_bit(y, 5) ^ get_bit(y, 7), 1);
    set_bit(&y, get_bit(y, 3) ^ get_bit(y, 6) ^ get_bit(y, 7), 2);
    set_bit(&y, get_bit(y, 5) ^ get_bit(y, 6) ^ get_bit(y, 7), 4);
    
    return y;
}

static void set_bit(uint8_t *x, _Bool bit, size_t i) {
    if (i == 0) {
        fprintf(stderr, "%s\n", "i = 0");
    }
    if (bit) {
        *x |= (1 << (i-1));
    } else {
        *x &= ~(1 << (i-1));
    }
}

static _Bool get_bit(uint8_t x, size_t i) {
    if (i == 0) {
        fprintf(stderr, "%s\n", "i = 0");
    }
    return (x & (1 << (i-1)) ? 1 : 0);
}

static void print_bits(uint8_t x) {
    for (size_t i = 1; i <= 8; ++i)
        printf("%d", get_bit(x, i));
    putchar('\n');
}

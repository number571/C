#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MODULO 64

typedef struct LFSR {
    uint64_t *blocks;
    uint64_t *ibits;
    size_t count;
    size_t size;
} LFSR;

extern LFSR *new_LFSR(size_t size, uint64_t *ibits);
extern void free_LFSR(LFSR *lfsr);

extern _Bool gamma_LFSR(LFSR *lfsr);

static _Bool _xor_bits(LFSR *lfsr);
static void _print_bits(uint64_t x, register uint64_t Nbit);

int main(void) {
    // uint64_t ibits[] = {
    //     0b1010000000000000000000000000000000000000000000000000000000000000,
    // };
    // LFSR *lfsr = new_LFSR(3, ibits);

    // _print_bits(lfsr->blocks[0], 64);
    // for (size_t i = 0; i < 8; ++i) {
    //     printf("%d", gamma_LFSR(lfsr));
    // }
    // putchar('\n');
    // _print_bits(lfsr->blocks[0], 64);

    // free_LFSR(lfsr);

    uint64_t ibits1[] = {
        0b1011000000000000000000000000000000000000000000000000000000000001,
    };
    LFSR *lfsr1 = new_LFSR(64, ibits1);

    uint64_t ibits2[] = {
        0b1100100000000000000000000000000000000000000000000000000000000000,
        0b0010000000000000000000000000000000000000000000000000000000000000,
    };
    LFSR *lfsr2 = new_LFSR(67, ibits2);

    uint64_t ibits3[] = {
        0b0111000000000000000000000000000000000000000000000000000000000000,
        0b0000000000000010000000000000000000000000000000000000000000000000,
    };
    LFSR *lfsr3 = new_LFSR(79, ibits3);

    for (size_t i = 0; i < 1024; ++i) {
        printf("%d", (gamma_LFSR(lfsr1) + gamma_LFSR(lfsr2) + gamma_LFSR(lfsr3) >= 2) ? 1 : 0); 
    }
    putchar('\n');

    free_LFSR(lfsr1);
    free_LFSR(lfsr2);
    free_LFSR(lfsr3);
    return 0;
}

extern LFSR *new_LFSR(size_t size, uint64_t *ibits) {
    LFSR *lfsr = (LFSR*)malloc(sizeof(LFSR));
    lfsr->size = size;
    lfsr->count = (size / MODULO) + (size % MODULO == 0 ? 0 : 1);

    lfsr->blocks = (uint64_t*)malloc(lfsr->count * sizeof(uint64_t));
    for (size_t i = 0; i < lfsr->count; ++i) {
        lfsr->blocks[i] = 0b0100000000000000000000000000000000000000000000000000000000000000;
    }

    if (lfsr->size % MODULO != 0) {
        lfsr->blocks[lfsr->count-1] >>= MODULO - (lfsr->size % MODULO);
    }

    lfsr->ibits = (uint64_t*)malloc(lfsr->count * sizeof(uint64_t));
    for (size_t i = 0; i < lfsr->count; ++i) {
        lfsr->ibits[i] = ibits[i];
    }

    if (lfsr->size % MODULO != 0) {
        lfsr->ibits[lfsr->count-1] >>= MODULO - (lfsr->size % MODULO);
    }

    return lfsr;
}

extern void free_LFSR(LFSR *lfsr) {
    free(lfsr->blocks);
    free(lfsr->ibits);
    free(lfsr);
}

extern _Bool gamma_LFSR(LFSR *lfsr) {
    _Bool gamma_bit = lfsr->blocks[lfsr->count-1] & 0x01;
    _Bool new_bit = _xor_bits(lfsr);
    _Bool carry = 0;
    _Bool temp = 0;
    uint64_t Nbit = MODULO;
    for (size_t i = 0; i < lfsr->count; ++i) {
        if (i == lfsr->count-1 && lfsr->size % MODULO != 0) {
            Nbit = lfsr->size % MODULO;
        }
        temp = lfsr->blocks[i] & 0x01;
        lfsr->blocks[i] >>= 1;
        lfsr->blocks[i] |= (uint64_t)carry << (Nbit-1);
        carry = temp;
    }
    if (lfsr->size < MODULO) {
        lfsr->blocks[0] |= (uint64_t)new_bit << (lfsr->size-1);
    } else {
        lfsr->blocks[0] |= (uint64_t)new_bit << (MODULO-1);
    }
    
    return gamma_bit;
}

static _Bool _xor_bits(LFSR *lfsr) {
    _Bool result = 0;
    for (size_t i = 0; i < lfsr->count; ++i) {
        uint64_t Nbit = MODULO;
        if (i == lfsr->count-1 && lfsr->size % MODULO != 0) {
            Nbit = lfsr->size % MODULO;
        }
        for (Nbit = (uint64_t)1 << (Nbit - 1); Nbit > 0x00; Nbit >>= 1) {
            if (lfsr->ibits[i] & Nbit) {
                result = result ^ (lfsr->blocks[i] & Nbit ? 1 : 0);
            }
        }
    }
    return result;
}

static void _print_bits(uint64_t x, register uint64_t Nbit) {
    for (Nbit = (uint64_t)1 << (Nbit - 1); Nbit > 0x00; Nbit >>= 1)
        printf("%d", (x & Nbit) ? 1 : 0);
    putchar('\n');
}

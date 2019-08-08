#include <stdio.h>
#include <stdint.h>

#define BUFF_SIZE 1024
#define LSHIFT_28BIT(x, L) ((((x) << (L)) | ((x) >> (-(L) & 27))) & (((uint64_t)1 << 32) - 1))

static const uint8_t __Sbox[8][4][16] = {
    { // 0
        {14, 4 , 13, 1 , 2 , 15, 11, 8 , 3 , 10, 6 , 12, 5 , 9 , 0 , 7 },
        {0 , 15, 7 , 4 , 14, 2 , 13, 1 , 10, 6 , 12, 11, 9 , 5 , 3 , 8 },
        {4 , 1 , 14, 8 , 13, 6 , 2 , 11, 15, 12, 9 , 7 , 3 , 10, 5 , 0 },
        {15, 12, 8 , 2 , 4 , 9 , 1 , 7 , 5 , 11, 3 , 14, 10, 0 , 6 , 13},
    },
    { // 1
        {15, 1 , 8 , 14, 6 , 11, 3 , 4 , 9 , 7 , 2 , 13, 12, 0 , 5 , 10},
        {3 , 13, 4 , 7 , 15, 2 , 8 , 14, 12, 0 , 1 , 10, 6 , 9 , 11, 5 },
        {0 , 14, 7 , 11, 10, 4 , 13, 1 , 5 , 8 , 12, 6 , 9 , 3 , 2 , 15},
        {13, 8 , 10, 1 , 3 , 15, 4 , 2 , 11, 6 , 7 , 12, 0 , 5 , 14, 9 },
    },
    { // 2
        {10, 0 , 9 , 14, 6 , 3 , 15, 5 , 1 , 13, 12, 7 , 11, 4 , 2 , 8 },
        {13, 7 , 0 , 9 , 3 , 4 , 6 , 10, 2 , 8 , 5 , 14, 12, 11, 15, 1 },
        {13, 6 , 4 , 9 , 8 , 15, 3 , 0 , 11, 1 , 2 , 12, 5 , 10, 14, 7 },
        {1 , 10, 13, 0 , 6 , 9 , 8 , 7 , 4 , 15, 14, 3 , 11, 5 , 2 , 12},
    },
    { // 3
        {7 , 13, 14, 3 , 0 , 6 , 9 , 10, 1 , 2 , 8 , 5 , 11, 12, 4 , 15},
        {13, 8 , 11, 5 , 6 , 15, 0 , 3 , 4 , 7 , 2 , 12, 1 , 10, 14, 9 },
        {10, 6 , 9 , 0 , 12, 11, 7 , 13, 15, 1 , 3 , 14, 5 , 2 , 8 , 4 },
        {3 , 15, 0 , 6 , 10, 1 , 13, 8 , 9 , 4 , 5 , 11, 12, 7 , 2 , 14},
    },
    { // 4
        {2 , 12, 4 , 1 , 7 , 10, 11, 6 , 8 , 5 , 3 , 15, 13, 0 , 14, 9 },
        {14, 11, 2 , 12, 4 , 7 , 13, 1 , 5 , 0 , 15, 10, 3 , 9 , 8 , 6 },
        {4 , 2 , 1 , 11, 10, 13, 7 , 8 , 15, 9 , 12, 5 , 6 , 3 , 0 , 14},
        {11, 8 , 12, 7 , 1 , 14, 2 , 13, 6 , 15, 0 , 9 , 10, 4 , 5 , 3 },
    },
    { // 5
        {12, 1 , 10, 15, 9 , 2 , 6 , 8 , 0 , 13, 3 , 4 , 14, 7 , 5 , 11},
        {10, 15, 4 , 2 , 7 , 12, 9 , 5 , 6 , 1 , 13, 14, 0 , 11, 3 , 8 },
        {9 , 14, 15, 5 , 2 , 8 , 12, 3 , 7 , 0 , 4 , 10, 1 , 13, 11, 6 },
        {4 , 3 , 2 , 12, 9 , 5 , 15, 10, 11, 14, 1 , 7 , 6 , 0 , 8 , 13},
    },
    { // 6
        {4 , 11, 2 , 14, 15, 0 , 8 , 13, 3 , 12, 9 , 7 , 5 , 10, 6 , 1 },
        {13, 0 , 11, 7 , 4 , 9 , 1 , 10, 14, 3 , 5 , 12, 2 , 15, 8 , 6 },
        {1 , 4 , 11, 13, 12, 3 , 7 , 14, 10, 15, 6 , 8 , 0 , 5 , 9 , 2 },
        {6 , 11, 13, 8 , 1 , 4 , 10, 7 , 9 , 5 , 0 , 15, 14, 2 , 3 , 12},
    },
    { // 7
        {13, 2 , 8 , 4 , 6 , 15, 11, 1 , 10, 9 , 3 , 14, 5 , 0 , 12, 7 },
        {1 , 15, 13, 8 , 10, 3 , 7 , 4 , 12, 5 , 6 , 11, 0 , 14, 9 , 2 },
        {7 , 11, 4 , 1 , 9 , 12, 14, 2 , 0 , 6 , 10, 13, 15, 3 , 5 , 8 },
        {2 , 1 , 14, 7 , 4 , 10, 8 , 13, 15, 12, 9 , 0 , 3 , 5 , 6 , 11},
    },
};

static const uint8_t __IP[64] = {
    58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9 , 1, 59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7,
};

static const uint8_t __FP[64] = {
    40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9 , 49, 17, 57, 25,
};

static const uint8_t __K1P[28] = {
    57, 49, 41, 33, 25, 17, 9 , 1 , 58, 50, 42, 34, 26, 18,
    10, 2 , 59, 51, 43, 35, 27, 19, 11, 3 , 60, 52, 44, 36,
};

static const uint8_t __K2P[28] = {
    63, 55, 47, 39, 31, 23, 15, 7 , 62, 54, 46, 38, 30, 22,
    14, 6 , 61, 53, 45, 37, 29, 21, 13, 5 , 28, 20, 12, 4 ,
};

static const uint8_t __CP[48] = {
    14, 17, 11, 24, 1 , 5 , 3 , 28, 15, 6 , 21, 10, 
    23, 19, 12, 4 , 26, 8 , 16, 7 , 27, 20, 13, 2 , 
    41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48, 
    44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32,
};

static const uint8_t __EP[48] = {
    32, 1 , 2 , 3 , 4 , 5 , 4 , 5 , 6 , 7 , 8 , 9 ,
    8 , 9 , 10, 11, 12, 13, 12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1 ,
};

static const uint8_t __P[32] = {
    16, 7 , 20, 21, 29, 12, 28, 17, 1 , 15, 23, 26, 5 , 18, 31, 10,
    2 , 8 , 24, 14, 32, 27, 3 , 9 , 19, 13, 30, 6 , 22, 11, 4 , 25,
};

size_t DES(uint8_t * to, uint8_t mode, uint8_t * keys8b, uint8_t * from, size_t length);

void key_expansion(uint64_t key64b, uint64_t * keys48b);
void key_permutation_56bits_to_28bits(uint64_t block56b, uint32_t * block32b_1, uint32_t * block32b_2);
void key_expansion_to_48bits(uint32_t block28b_1, uint32_t block28b_2, uint64_t * keys48b);
uint64_t key_contraction_permutation(uint64_t block56b);

void feistel_cipher(uint8_t mode, uint32_t * N1, uint32_t * N2, uint64_t * keys48b);
void round_feistel_cipher(uint32_t * N1, uint32_t * N2, uint64_t key48b);
uint32_t func_F(uint32_t block32b, uint64_t key48b);
uint64_t expansion_permutation(uint32_t block32b);
uint32_t substitutions(uint64_t block48b);
void substitution_6bits_to_4bits(uint8_t * blocks6b, uint8_t * blocks4b);
uint32_t permutation(uint32_t block32b);

uint8_t extreme_bits(uint8_t block6b);
uint8_t middle_bits(uint8_t block6b);

uint64_t initial_permutation(uint64_t block64b);
uint64_t final_permutation(uint64_t block64b);

void split_64bits_to_32bits(uint64_t block64b, uint32_t * block32b_1, uint32_t * block32b_2);
void split_64bits_to_8bits(uint64_t block64b, uint8_t * blocks8b);
void split_48bits_to_6bits(uint64_t block48b, uint8_t * blocks6b);

uint64_t join_32bits_to_64bits(uint32_t block32b_1, uint32_t block32b_2);
uint64_t join_28bits_to_56bits(uint32_t block28b_1, uint32_t block28b_2);
uint64_t join_8bits_to_64bits(uint8_t * blocks8b);
uint32_t join_4bits_to_32bits(uint8_t * blocks8b);

static inline size_t input_string(uint8_t * buffer);
static inline void swap(uint32_t * N1, uint32_t * N2);
static inline void print_array(uint8_t * array, size_t length);
static inline void print_bits(uint64_t x, register uint64_t Nbit);

int main(void) {
    uint8_t encrypted[BUFF_SIZE], decrypted[BUFF_SIZE];
    uint8_t buffer[BUFF_SIZE] = {0};
    uint8_t keys8b[8] = "DESkey56";

    size_t length = input_string(buffer);
    print_array(buffer, length);

    length = DES(encrypted, 'E', keys8b, buffer, length);
    print_array(encrypted, length);

    length = DES(decrypted, 'D', keys8b, encrypted, length);
    print_array(decrypted, length);

    return 0;
}

size_t DES(uint8_t * to, uint8_t mode, uint8_t * keys8b, uint8_t * from, size_t length) {
    length = length % 8 == 0 ? length : length + (8 - (length % 8));
    
    uint64_t keys48b[16] = {0};
    uint32_t N1, N2;

    key_expansion(
        join_8bits_to_64bits(keys8b), 
        keys48b
    );

    for (size_t i = 0; i < length; i += 8) {
        split_64bits_to_32bits(
            initial_permutation(
                join_8bits_to_64bits(from + i)
            ), 
            &N1, &N2
        );
        feistel_cipher(mode, &N1, &N2, keys48b);
        split_64bits_to_8bits(
            final_permutation(
                join_32bits_to_64bits(N1, N2)
            ),
            (to + i)
        );
    }

    return length;
}

void feistel_cipher(uint8_t mode, uint32_t * N1, uint32_t * N2, uint64_t * keys48b) {
    switch(mode) {
        case 'E': case 'e': {
            for (int8_t round = 0; round < 16; ++round) {
                round_feistel_cipher(N1, N2, keys48b[round]);
            }
            swap(N1, N2);
            break;
        }
        case 'D': case 'd': {
            for (int8_t round = 15; round >= 0; --round) {
                round_feistel_cipher(N1, N2, keys48b[round]);
            }
            swap(N1, N2);
            break;
        }
    }
}

void round_feistel_cipher(uint32_t * N1, uint32_t * N2, uint64_t key48b) {
    uint32_t temp = *N2;
    *N2 = func_F(*N2, key48b) ^ *N1;
    *N1 = temp;
}

uint32_t func_F(uint32_t block32b, uint64_t key48b) {
    uint64_t block48b = expansion_permutation(block32b);
    block48b ^= key48b;
    block32b = substitutions(block48b);
    return permutation(block32b);
}

uint64_t expansion_permutation(uint32_t block32b) {
    uint64_t block48b = 0;
    for (uint8_t i = 0 ; i < 48; ++i) {
        block48b |= (uint64_t)((block32b >> (32 - __EP[i])) & 0x01) << (63 - i);
    }
    return block48b;
}

uint32_t substitutions(uint64_t block48b) {
    uint8_t blocks4b[4], blocks6b[8] = {0};
    split_48bits_to_6bits(block48b, blocks6b);
    substitution_6bits_to_4bits(blocks6b, blocks4b);
    return join_4bits_to_32bits(blocks4b);
}

void substitution_6bits_to_4bits(uint8_t * blocks6b, uint8_t * blocks4b) {
    uint8_t block2b, block4b;

    for (uint8_t i = 0, j = 0; i < 8; i += 2, ++j) {
        block2b = extreme_bits(blocks6b[i]);
        block4b = middle_bits(blocks6b[i]);
        blocks4b[j] = __Sbox[i][block2b][block4b];

        block2b = extreme_bits(blocks6b[i+1]);
        block4b = middle_bits(blocks6b[i+1]);
        blocks4b[j] = (blocks4b[j] << 4) | __Sbox[i+1][block2b][block4b];
    }
}

uint8_t extreme_bits(uint8_t block6b) {
    return ((block6b >> 6) & 0x2) | ((block6b >> 2) & 0x1);
}

uint8_t middle_bits(uint8_t block6b) {
    return (block6b >> 3) & 0xF;
}

uint32_t permutation(uint32_t block32b) {
    uint32_t new_block32b = 0;
    for (uint8_t i = 0 ; i < 32; ++i) {
        new_block32b |= ((block32b >> (32 - __P[i])) & 0x01) << (31 - i);
    }
    return new_block32b;
}

uint64_t initial_permutation(uint64_t block64b) {
    uint64_t new_block64b = 0;
    for (uint8_t i = 0 ; i < 64; ++i) {
        new_block64b |= ((block64b >> (64 - __IP[i])) & 0x01) << (63 - i);
    }
    return new_block64b;
}

uint64_t final_permutation(uint64_t block64b) {
    uint64_t new_block64b = 0;
    for (uint8_t i = 0 ; i < 64; ++i) {
        new_block64b |= ((block64b >> (64 - __FP[i])) & 0x01) << (63 - i);
    }
    return new_block64b;
}

void key_expansion(uint64_t key64b, uint64_t * keys48b) {
    uint32_t K1 = 0, K2 = 0;
    key_permutation_56bits_to_28bits(key64b, &K1, &K2);
    key_expansion_to_48bits(K1, K2, keys48b);
}

void key_permutation_56bits_to_28bits(uint64_t block56b, uint32_t * block28b_1, uint32_t * block28b_2) {
    for (uint8_t i = 0; i < 28; ++i) {
        *block28b_1 |= ((block56b >> (64 - __K1P[i])) & 0x01) << (31 - i);
        *block28b_2 |= ((block56b >> (64 - __K2P[i])) & 0x01) << (31 - i);
    }
}

void key_expansion_to_48bits(uint32_t block28b_1, uint32_t block28b_2, uint64_t * keys48b) {
    uint64_t block56b;
    uint8_t n;

    for (uint8_t i = 0; i < 16; ++i) {
        switch(i) {
            case 0: case 1: case 8: case 15: n = 1; break;
            default: n = 2; break;
        }

        block28b_1 = LSHIFT_28BIT(block28b_1, n);
        block28b_2 = LSHIFT_28BIT(block28b_2, n);
        block56b = join_28bits_to_56bits(block28b_1, block28b_2);
        keys48b[i] = key_contraction_permutation(block56b);
    }
}

uint64_t key_contraction_permutation(uint64_t block56b) {
    uint64_t block48b = 0;
    for (uint8_t i = 0 ; i < 48; ++i) {
        block48b |= ((block56b >> (64 - __CP[i])) & 0x01) << (63 - i);
    }
    return block48b;
}

void split_64bits_to_32bits(uint64_t block64b, uint32_t * block32b_1, uint32_t * block32b_2) {
    *block32b_1 = (uint32_t)(block64b >> 32);
    *block32b_2 = (uint32_t)(block64b);
}

void split_64bits_to_8bits(uint64_t block64b, uint8_t * blocks8b) {
    for (size_t i = 0; i < 8; ++i) {
        blocks8b[i] = (uint8_t)(block64b >> ((7 - i) * 8));
    }
}

void split_48bits_to_6bits(uint64_t block48b, uint8_t * blocks6b) {
    for (uint8_t i = 0; i < 8; ++i) {
        blocks6b[i] = (block48b >> (58 - (i * 6))) << 2;
    }
}

uint64_t join_32bits_to_64bits(uint32_t block32b_1, uint32_t block32b_2) {
    uint64_t block64b;
    block64b = (uint64_t)block32b_1;
    block64b = (uint64_t)(block64b << 32) | block32b_2;
    return block64b;
}

uint64_t join_28bits_to_56bits(uint32_t block28b_1, uint32_t block28b_2) {
    uint64_t block56b;
    block56b = (block28b_1 >> 4);
    block56b = ((block56b << 32) | block28b_2) << 4;
    return block56b;
}

uint64_t join_8bits_to_64bits(uint8_t * blocks8b) {
    uint64_t block64b;
    for (uint8_t *p = blocks8b; p < blocks8b + 8; ++p) {
        block64b = (block64b << 8) | *p;
    }
    return block64b;
}

uint32_t join_4bits_to_32bits(uint8_t * blocks4b) {
    uint32_t block32b;
    for (uint8_t *p = blocks4b; p < blocks4b + 4; ++p) {
        block32b = (block32b << 8) | *p;
    }
    return block32b;
}

static inline size_t input_string(uint8_t * buffer) {
    size_t position = 0;
    uint8_t ch;
    while ((ch = getchar()) != '\n' && position < BUFF_SIZE - 1)
        buffer[position++] = ch;
    buffer[position] = '\0';
    return position;
}

static inline void swap(uint32_t * N1, uint32_t * N2) {
    uint32_t temp = *N1;
    *N1 = *N2;
    *N2 = temp;
}

static inline void print_array(uint8_t * array, size_t length) {
    printf("[ ");
    for (size_t i = 0; i < length; ++i)
        printf("%d ", array[i]);
    printf("]\n");
}

static inline void print_bits(uint64_t x, register uint64_t Nbit) {
    for (Nbit = (uint64_t)1 << (Nbit - 1); Nbit > 0x00; Nbit >>= 1)
        printf("%d", (x & Nbit) ? 1 : 0);
    putchar('\n');
}

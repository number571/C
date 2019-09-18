#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define BUFF_SIZE 1024

const uint8_t Sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, 
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, 
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
};

const uint8_t InvSbox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
};

const uint32_t Rcon[11] = {
    0x00000000,
    0x01000000,
    0x02000000,
    0x04000000,
    0x08000000,
    0x10000000,
    0x20000000,
    0x40000000,
    0x80000000,
    0x1B000000,
    0x36000000,
};

size_t AES(uint8_t * to, uint8_t mode, uint8_t * key, uint8_t key_size, uint8_t * from, size_t length);
void key_expansion(uint32_t * Wkey, uint8_t * key, uint8_t Nb, uint8_t Nk, uint8_t Nr);

void inv_sub_bytes(uint8_t * block);
void sub_bytes(uint8_t * block);

void inv_shift_rows(uint8_t * block);
void shift_rows(uint8_t * block);

void inv_mix_columns(uint8_t * block);
void mix_columns(uint8_t * block);

void add_round_key(uint8_t * block, uint32_t * Wkeys);
uint8_t GF_mul(uint8_t a, uint8_t b);

void shiftr_array(uint8_t * array, size_t length, size_t shift);
void shiftl_array(uint8_t * array, size_t length, size_t shift);

uint32_t sub_word(uint32_t word);
uint32_t rot_word(uint32_t word);

void copy_transpose_block(uint8_t * to, uint8_t * from);
void copy_block(uint8_t * to, uint8_t * from);

uint32_t join_8bits_to_32bits(uint8_t * blocks8b);
void split_32bits_to_8bits(uint32_t block32b, uint8_t * blocks8b);

static inline size_t input_string(uint8_t * buffer);
static inline void print_bytes(uint8_t * array, size_t length);
static inline void print_array(uint8_t * array, size_t length);
static inline void print_bits(uint64_t x, register uint64_t Nbit);

#define KEY_SIZE 16

int main(void) {
    uint8_t encrypted[BUFF_SIZE] = {0};
    uint8_t decrypted[BUFF_SIZE] = {0};
    uint8_t buffer[BUFF_SIZE];
    uint8_t key[KEY_SIZE] = "AES_key_128_bits";

    size_t length = input_string(buffer);
    print_bytes(buffer, length);

    length = AES(encrypted, 'E', key, KEY_SIZE, buffer, length);
    print_bytes(encrypted, length);

    length = AES(decrypted, 'D', key, KEY_SIZE, encrypted, length);
    print_bytes(decrypted, length);
    return 0;
}

size_t AES(uint8_t * to, uint8_t mode, uint8_t * key, uint8_t key_size, uint8_t * from, size_t length) {
    if ((key_size != 16 && key_size != 24 && key_size != 32) || (mode != 'E' && mode != 'D')) {
        return -1;
    }

    length = (length % key_size == 0) ? length : length + (key_size - (length % key_size));

    const uint8_t Nb = 4;
    const uint8_t Nk = key_size / Nb;
    const uint8_t Nr = (key_size == 16) ? 10 : (key_size == 24) ? 12 : 14;
    const size_t  Nw = Nb * (Nr + 1);

    uint32_t Wkey[Nw];
    key_expansion(Wkey, key, Nb, Nk, Nr);

    uint8_t block[16];
    
    switch(mode) {
        case 'E': {
            for (size_t i = 0; i < length; i += 16) {
                copy_transpose_block(block, from + i);
                add_round_key(block, Wkey);
                for (uint8_t round = 1; round <= Nr; ++round) {
                    sub_bytes(block);
                    shift_rows(block);
                    if (round < Nr) {
                        mix_columns(block);
                    }
                    add_round_key(block, Wkey + (Nb * round));
                }
                copy_transpose_block(to + i, block);
            }
        }
        break;
        case 'D': {
            for (size_t i = 0; i < length; i += 16) {
                copy_transpose_block(block, from + i);
                add_round_key(block, Wkey + (Nb * Nr));
                for (int8_t round = Nr - 1; round >= 0; --round) {
                    inv_shift_rows(block);
                    inv_sub_bytes(block);
                    add_round_key(block, Wkey + (Nb * round));
                    if (round > 0) {
                        inv_mix_columns(block);
                    }
                }
                copy_transpose_block(to + i, block);
            }
        }
        break;
    }
    
    return length;
}

void key_expansion(uint32_t * Wkey, uint8_t * key, uint8_t Nb, uint8_t Nk, uint8_t Nr) {
    for (uint8_t i = 0; i < Nk; ++i) {
        Wkey[i] = join_8bits_to_32bits(key + (4 * i));
    }
    uint32_t T;
    for (size_t i = Nk; i < (Nb * (Nr + 1)); ++i) {
        T = Wkey[i-1];
        if (i % Nk == 0) {
            T = sub_word(rot_word(T)) ^ Rcon[i/Nk];
        } else if (Nk == 8 && (i % Nk) == 4) {
            T = sub_word(T);
        } 
        Wkey[i] = Wkey[i-Nk] ^ T;
    }
}

uint32_t sub_word(uint32_t word) {
    uint8_t bytes[4];
    split_32bits_to_8bits(word, bytes);
    for (uint8_t i = 0; i < 4; ++i) {
        bytes[i] = Sbox[bytes[i]];
    }
    return join_8bits_to_32bits(bytes);
}

uint32_t rot_word(uint32_t word) {
    uint8_t bytes[4];
    split_32bits_to_8bits(word, bytes);
    shiftl_array(bytes, 4, 1);
    return join_8bits_to_32bits(bytes);
}

void copy_transpose_block(uint8_t * to, uint8_t * from) {
    for (uint8_t i = 0; i < 4; ++i) {
        for (uint8_t j = 0; j < 4; ++j) {
            to[i * 4 + j] = from[j * 4 + i];
        }
    }
}

void copy_block(uint8_t * to, uint8_t * from) {
    for (uint8_t i = 0; i < 4; ++i) {
        for (uint8_t j = 0; j < 4; ++j) {
            to[i * 4 + j] = from[i * 4 + j];
        }
    }
}

void inv_sub_bytes(uint8_t * block) {
    for (uint8_t i = 0; i < 16; ++i) {
        block[i] = InvSbox[block[i]];
    }
}

void sub_bytes(uint8_t * block) {
    for (uint8_t i = 0; i < 16; ++i) {
        block[i] = Sbox[block[i]];
    }
}

void inv_shift_rows(uint8_t * block) {
    for (uint8_t i = 1; i < 4; ++i) {
        shiftr_array(block + (i * 4), 4, i);
    }
}

void shift_rows(uint8_t * block) {
    for (uint8_t i = 1; i < 4; ++i) {
        shiftl_array(block + (i * 4), 4, i);
    }
}

// [14 11 13 9]
// [9 14 11 13]
// [13 9 14 11]
// [11 13 9 14]
void inv_mix_columns(uint8_t * block) {
    uint8_t columns[16];
    copy_block(columns, block);
    for (uint8_t i = 0; i < 4; ++i) {
        block[0*4+i] = GF_mul(0x0E, columns[0*4+i]) ^ GF_mul(0x0B, columns[1*4+i]) ^ GF_mul(0x0D, columns[2*4+i]) ^ GF_mul(0x09, columns[3*4+i]);
        block[1*4+i] = GF_mul(0x09, columns[0*4+i]) ^ GF_mul(0x0E, columns[1*4+i]) ^ GF_mul(0x0B, columns[2*4+i]) ^ GF_mul(0x0D, columns[3*4+i]);
        block[2*4+i] = GF_mul(0x0D, columns[0*4+i]) ^ GF_mul(0x09, columns[1*4+i]) ^ GF_mul(0x0E, columns[2*4+i]) ^ GF_mul(0x0B, columns[3*4+i]);
        block[3*4+i] = GF_mul(0x0B, columns[0*4+i]) ^ GF_mul(0x0D, columns[1*4+i]) ^ GF_mul(0x09, columns[2*4+i]) ^ GF_mul(0x0E, columns[3*4+i]);
    }
}

// [2 3 1 1]
// [1 2 3 1]
// [1 1 2 3]
// [3 1 1 2]
void mix_columns(uint8_t * block) {
    uint8_t columns[16];
    copy_block(columns, block);
    for (uint8_t i = 0; i < 4; ++i) {
        block[0*4+i] = GF_mul(0x02, columns[0*4+i]) ^ GF_mul(0x03, columns[1*4+i]) ^ columns[2*4+i] ^ columns[3*4+i];
        block[1*4+i] = columns[0*4+i] ^ GF_mul(0x02, columns[1*4+i]) ^ GF_mul(0x03, columns[2*4+i]) ^ columns[3*4+i];
        block[2*4+i] = columns[0*4+i] ^ columns[1*4+i] ^ GF_mul(0x02, columns[2*4+i]) ^ GF_mul(0x03, columns[3*4+i]);
        block[3*4+i] = GF_mul(0x03, columns[0*4+i]) ^ columns[1*4+i] ^ columns[2*4+i] ^ GF_mul(0x02, columns[3*4+i]);
    }
}

uint8_t GF_mul(uint8_t a, uint8_t b) {
    uint8_t hi_bit_set, p = 0;
    for(uint8_t counter = 0; counter < 8; counter++) {
        if(b & 0x01) {
            p ^= a;
        }
        hi_bit_set = (a & 0x80);
        a <<= 1;
        if(hi_bit_set == 0x80)  {
            a ^= 0x1B;      
        }
        b >>= 1;
    }
    return p;
}

void add_round_key(uint8_t * block, uint32_t * Wkeys) {
    uint8_t bytes[4];
    for (uint8_t i = 0; i < 4; ++i) {
        split_32bits_to_8bits(Wkeys[i], bytes);
        for (uint8_t j = 0; j < 4; ++j) {
            block[j * 4 + i] ^= bytes[j];
        }
    }
}

void shiftr_array(uint8_t * array, size_t length, size_t shift) {
    uint8_t temp;
    size_t index;
    while (shift--) {
        temp = array[length-1];
        for (index = length-1; index > 0; index--)
            array[index] = array[index-1];
        array[0] = temp;
    }
}

void shiftl_array(uint8_t * array, size_t length, size_t shift) {
    uint8_t temp;
    size_t index;
    while(shift--) {
        temp = array[0];
        index = 1;
        for (; index < length; ++index) {
            array[index-1] = array[index];
        }
        array[index-1] = temp;
    }
}

uint32_t join_8bits_to_32bits(uint8_t * blocks8b) {
    uint64_t block32b;
    for (uint8_t *p = blocks8b; p < blocks8b + 4; ++p) {
        block32b = (block32b << 8) | *p;
    }
    return block32b;
}

void split_32bits_to_8bits(uint32_t block32b, uint8_t * blocks8b) {
    for (uint8_t i = 0; i < 4; ++i) {
        blocks8b[i] = (uint8_t)(block32b >> (24 - i * 8));
    }
}

static inline size_t input_string(uint8_t * buffer) {
    size_t position = 0;
    uint8_t ch;
    while ((ch = getchar()) != '\n' && position < BUFF_SIZE - 1)
        buffer[position++] = ch;
    buffer[position] = '\0';
    return position;
}

static inline void print_bytes(uint8_t * array, size_t length) {
    printf("[ ");
    for (size_t i = 0; i < length; ++i)
        printf("%x ", array[i]);
    printf("]\n");
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

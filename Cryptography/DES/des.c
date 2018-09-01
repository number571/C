#include "library.h"

static const unsigned char initial_message_permutation[64] = {
    58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17,  9, 1, 59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7
};

static const unsigned char final_message_permutation[64] = {
    40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41,  9, 49, 17, 57, 25
};

static const unsigned char message_expansion[48] = {
    32,  1,  2,  3,  4,  5,
     4,  5,  6,  7,  8,  9,
     8,  9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32,  1
};

static const unsigned char S1[64] = {
    14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
    0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
    4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
    15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13
};

static const unsigned char S2[64] = {
    15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
    3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
    0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
    13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9
};

static const unsigned char S3[64] = {
    10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
    13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
    13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
    1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12
};

static const unsigned char S4[64] = {
    7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
    13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
    10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
    3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14
};

static const unsigned char S5[64] = {
    2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
    14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
    4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
    11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3
};

static const unsigned char S6[64] = {
    12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
    10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
    9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
    4, 3, 2, 12, 9, 5, 15, 10, 11,14, 1, 7, 6, 0, 8, 13
};

static const unsigned char S7[64] = {
    4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
    13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
    1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
    6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12
};

static const unsigned char S8[64] = {
    13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
    1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
    7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
    2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11
};

static const unsigned char right_sub_message_permutation[32] = {
    16,  7, 20, 21,
    29, 12, 28, 17,
     1, 15, 23, 26,
     5, 18, 31, 10,
     2,  8, 24, 14,
    32, 27,  3,  9,
    19, 13, 30,  6,
    22, 11,  4, 25
};

static const unsigned char initial_key_permutaion[56] = {
    57, 49, 41, 33, 25, 17,  9,  1, 58, 50, 42, 34, 26, 18,
    10,  2, 59, 51, 43, 35, 27, 19, 11,  3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,  7, 62, 54, 46, 38, 30, 22,
    14,  6, 61, 53, 45, 37, 29, 21, 13,  5, 28, 20, 12,  4
};

static const unsigned char key_shift_sizes[16] = {1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1};

static const unsigned char sub_key_permutation[48] = {
    14, 17, 11, 24,  1,  5,  3, 28, 15,  6, 21, 10, 23, 19, 12,  4, 
    26,  8, 16,  7, 27, 20, 13,  2, 41, 52, 31, 37, 47, 55, 30, 40, 
    51, 45, 33, 48, 44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32
};

static unsigned char shift (
    const unsigned char shift_size, 
    const unsigned char n, 
    const unsigned char* const message
) {
    unsigned char shift_byte;
    shift_byte = 0x80 >> ((shift_size - n) % DES_BLOCK);
    shift_byte &= message[(shift_size - n) / DES_BLOCK];
    shift_byte <<= ((shift_size - n) % DES_BLOCK);
    return shift_byte;
}

static void set (
    const unsigned char shift_size, 
    const unsigned char shift_byte, 
    unsigned char shift_bits[], 
    unsigned char key[]
) {
    unsigned i;
    for (i = 0; i < 4; i++) 
        shift_bits[i] = shift_byte & key[i];
    for (i = 1; i < 4; i++) {
        key[i-1] <<= shift_size;
        key[i-1] |= (shift_bits[i] >> (DES_BLOCK - shift_size));
    }
    key[3] <<= shift_size;
    key[3] |= (shift_bits[0] >> (4 - shift_size));
}

extern void generate_keys (const unsigned char* const main_key, key_set* key_sets) {
    unsigned char shift_bits[4], shift_byte, shift_size;
    unsigned int i, j;

    memset(key_sets[0].k, 0, DES_BLOCK);

    for (i = 0; i < 56; i++) {
        shift_size = initial_key_permutaion[i];
        key_sets[0].k[i/DES_BLOCK] |= (shift(shift_size, 1, main_key) >> i % DES_BLOCK);
    }

    for (i = 0; i < 3; i++) 
        key_sets[0].c[i] = key_sets[0].k[i];

    key_sets[0].c[3] = key_sets[0].k[3] & 0xF0;

    for (i = 0; i < 3; i++) {
        key_sets[0].d[i] = (key_sets[0].k[i+3] & 0x0F) << 4;
        key_sets[0].d[i] |= (key_sets[0].k[i+4] & 0xF0) >> 4;
    }

    key_sets[0].d[3] = (key_sets[0].k[6] & 0x0F) << 4;

    for (i = 0; i < 16; i++) {

        for (j = 0; j < 4; j++) {
            key_sets[i+1].c[j] = key_sets[i].c[j];
            key_sets[i+1].d[j] = key_sets[i].d[j];
        }

        shift_size = key_shift_sizes[i];
        set(shift_size, 0x80, shift_bits, key_sets[i+1].c);
        set(shift_size, 0x80, shift_bits, key_sets[i+1].d);

        for (j = 0; j < 48; j++) {
            shift_size = sub_key_permutation[j];

            if (shift_size <= 28) 
                shift_byte = shift(shift_size, 1, key_sets[i+1].c);
            else 
                shift_byte = shift(shift_size, 29, key_sets[i+1].d);

            key_sets[i+1].k[j/DES_BLOCK] |= (shift_byte >> j % DES_BLOCK);
        }
    }
}

static void transposition (
    const _Bool tr, 
    const unsigned char* const message, 
    unsigned char permutation[]
) {
    unsigned char i, shift_size;
    for (i = 0; i < 64; i++) {
        if (!tr) 
            shift_size = initial_message_permutation[i];
        else 
            shift_size = final_message_permutation[i];
        permutation[i/DES_BLOCK] |= (shift(shift_size, 1, message) >> i % DES_BLOCK);
    }
}

extern void feistel_function (
    const unsigned char* const data_block, 
    unsigned char processed_piece[], 
    key_set* key_sets, 
    const unsigned char mode
) {
    unsigned char i, j, shift_size, row, column, key_index;
    unsigned char begin[DES_BLOCK], end[DES_BLOCK];
    unsigned char l[4], r[4], ln[4], rn[4], er[6], ser[4];

    memset(begin, 0, DES_BLOCK);
    memset(processed_piece, 0, DES_BLOCK);

    transposition(0, data_block, begin);

    for (i = 0; i < 4; i++) {
        l[i] = begin[i];
        r[i] = begin[i+4];
    }

    for (i = 0; i < 16; i++) {
        memcpy(ln, r, 4);
        memset(er, 0, 6);

        for (j = 0; j < 48; j++) {
            shift_size = message_expansion[j];
            er[j/DES_BLOCK] |= (shift(shift_size, 1, r) >> j % DES_BLOCK);
        }

        key_index = (mode == ENCRYPTION_MODE) ? (i + 1) : (16 - i); 

        for (j = 0; j < 6; j++) 
            er[j] ^= key_sets[key_index].k[j];

        memset(ser, 0, 4);

        row = column = 0;
        row |= ((er[0] & 0x80) >> 6) | ((er[0] & 0x04) >> 2);
        column |= ((er[0] & 0x78) >> 3);
        ser[0] |= (S1[row*16+column] << 4);

        row = column = 0;
        row |= (er[0] & 0x02) | ((er[1] & 0x10) >> 4);
        column |= ((er[0] & 0x01) << 3) | ((er[1] & 0xE0) >> 5);
        ser[0] |= S2[row*16+column];

        row = column = 0;
        row |= ((er[1] & 0x08) >> 2) | ((er[2] & 0x40) >> 6);
        column |= ((er[1] & 0x07) << 1) | ((er[2] & 0x80) >> 7);
        ser[1] |= (S3[row*16+column] << 4);

        row = column = 0;
        row |= ((er[2] & 0x20) >> 4) | (er[2] & 0x01);
        column |= ((er[2] & 0x1E) >> 1);
        ser[1] |= S4[row*16+column];

        row = column = 0;
        row |= ((er[3] & 0x80) >> 6) | ((er[3] & 0x04) >> 2);
        column |= ((er[3] & 0x78) >> 3);
        ser[2] |= (S5[row*16+column] << 4);

        row = column = 0;
        row |= (er[3] & 0x02) | ((er[4] & 0x10) >> 4);
        column |= ((er[3] & 0x01) << 3) | ((er[4] & 0xE0) >> 5);
        ser[2] |= S6[row*16+column];

        row = column = 0;
        row |= ((er[4] & 0x08) >> 2) | ((er[5] & 0x40) >> 6);
        column |= ((er[4] & 0x07) << 1) | ((er[5] & 0x80) >> 7);
        ser[3] |= (S7[row*16+column] << 4);

        row = column = 0;
        row |= ((er[5] & 0x20) >> 4) | (er[5] & 0x01);
        column |= ((er[5] & 0x1E) >> 1);
        ser[3] |= S8[row*16+column];

        memset(rn, 0, 4);

        for (j = 0; j < 32; j++) {
            shift_size = right_sub_message_permutation[j];
            rn[j/DES_BLOCK] |= (shift(shift_size, 1, ser) >> j % DES_BLOCK);
        }

        for (j = 0; j < 4; j++) rn[j] ^= l[j];
        for (j = 0; j < 4; j++) {
            l[j] = ln[j]; 
            r[j] = rn[j];
        }
    }

    for (j = 0; j < 4; j++) {
        end[j] = r[j];
        end[4+j] = l[j];
    }

    transposition(1, end, processed_piece);
}

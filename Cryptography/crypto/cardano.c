#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define ENCRYPT_MODE  1
#define DECRYPT_MODE -1

#define END_OF_STRING '\0'
#define SQUARE(x) ((x) * (x))

static void rotate90(uint8_t * const matrix, const size_t length);
static void print_matrix(const uint8_t * const matrix, const size_t length);
static void encrypt(
    uint8_t * const to,
    const uint8_t const key[][2],
    const size_t key_length,
    const size_t matrix_length,
    const uint8_t * const from,
    const size_t length
);
static void decrypt(
    uint8_t * const to,
    const uint8_t const key[][2],
    const size_t key_length,
    const size_t matrix_length,
    const uint8_t * const from,
    const size_t length
);
extern char cardano(
    uint8_t * const to,
    const int8_t mode,
    const uint8_t const key[][2],
    const size_t key_length,
    const size_t matrix_length,
    const uint8_t * const from,
    const size_t length
);

int main(void) {
    uint8_t message[26] = "SECRET";
    uint8_t key[][2] = { {2, 1}, {2, 4}, {3, 3} };

    const size_t length = strlen(message);
    const size_t matrix_length = 5;
    const size_t key_length = 3;

    cardano(message, ENCRYPT_MODE, key, key_length, matrix_length, message, length);
    print_matrix(message, matrix_length);
    printf("%s\n", message);
    return 0;
}

static void print_matrix(const uint8_t * const matrix, const size_t length) {
    for (size_t i = 0; i < length; ++i) {
        for (size_t j = 0; j < length; ++j) {
            printf("%c ", matrix[i * length + j]);
        }
        putchar('\n');
    }
}

static void rotate90(uint8_t * const matrix, const size_t length) {
    const size_t size = SQUARE(length);
    uint8_t rotated[size];
    for (size_t i = 0; i < length; ++i) {
        for (size_t j = 0; j < length; ++j) {
            rotated[length * j + i] = matrix[length * (length-i-1) + j];
        }
    }
    memcpy(matrix, rotated, size);
}

static void encrypt(
    uint8_t * const to,
    const uint8_t const key[][2],
    const size_t key_length,
    const size_t matrix_length,
    const uint8_t * const from,
    const size_t length
) {
    const size_t matrix_size = SQUARE(matrix_length);
    uint8_t buffer[matrix_size];
    size_t position = 0;

    memcpy(buffer, from, matrix_size);

    srand(time(NULL));
    for (size_t i = 0; i < matrix_size; ++i) {
        buffer[i] = rand() % 26 + 65;
    }

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < key_length; ++j) {
            if (position == length) {
                break;
            } 
            buffer[key[j][0] * matrix_length + key[j][1]] = from[position++];
        }
        rotate90(buffer, matrix_length);
    }

    memcpy(to, buffer, matrix_size);
}

static void decrypt(
    uint8_t * const to,
    const uint8_t const key[][2],
    const size_t key_length,
    const size_t matrix_length,
    const uint8_t * const from,
    const size_t length
) {
    const size_t matrix_size = SQUARE(matrix_length);
    uint8_t buffer[matrix_size];
    size_t position = 0;

    memcpy(buffer, from, matrix_size);

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < key_length; ++j) {
            if (position == length) {
                goto end_decrypt;
            } 
            to[position++] = buffer[key[j][0] * matrix_length + key[j][1]];
        }
        rotate90(buffer, matrix_length);
    }

end_decrypt:
    to[position] = END_OF_STRING;
}

extern char cardano(
    uint8_t * const to,
    const int8_t mode,
    const uint8_t const key[][2],
    const size_t key_length,
    const size_t matrix_length,
    const uint8_t * const from,
    const size_t length
) {
    switch (mode) {
        case ENCRYPT_MODE: 
            encrypt(to, key, key_length, matrix_length, from, length);
        break;
        case DECRYPT_MODE:
            decrypt(to, key, key_length, matrix_length, from, length);
        break;
        default: return 1;
    }

    return 0;
}

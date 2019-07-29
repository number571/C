#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define ENCRYPT_MODE  1
#define DECRYPT_MODE -1

static void rotate90(uint8_t * matrix, size_t length);
static void print_matrix(uint8_t * matrix, size_t length);
extern char cardano(
    uint8_t * to,
    const int8_t mode,
    uint8_t coord[][2],
    size_t coord_len,
    size_t matrix_len,
    uint8_t * from,
    size_t length
);

int main(void) {
    uint8_t message[20] = "SECRET";
    uint8_t coord[][2] = {
        {0, 1},
        {1, 0},
    };
    cardano(message, ENCRYPT_MODE, coord, 2, 4, message, strlen(message));
    print_matrix(message, 4);

    return 0;
}

static void print_matrix(uint8_t * matrix, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        for (size_t j = 0; j < length; ++j) {
            printf("%c", matrix[i * length + j]);
        }
        putchar('\n');
    }
}

static void rotate90(uint8_t * matrix, size_t length) {
    const size_t size = length * length;
    uint8_t rotated[size];
    for (size_t i = 0; i < length; ++i) {
        for (size_t j = 0; j < length; ++j) {
            rotated[length * j + i] = matrix[length * (length-i-1) + j];
        }
    }
    memcpy(matrix, rotated, size);
}

extern char cardano(
    uint8_t * to,
    const int8_t mode,
    uint8_t coord[][2],
    size_t coord_len,
    size_t matrix_len,
    uint8_t * from,
    size_t length
) {
    if (mode != ENCRYPT_MODE && mode != DECRYPT_MODE) {
        return 1;
    }

    const size_t size = matrix_len * matrix_len;
    uint8_t buffer[size];
    memcpy(buffer, from, size);

    size_t position = 0;

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < coord_len; ++j) {
            buffer[coord[j][0] * length + coord[j][1]] = from[position++];
            if (position == length) {
                goto end_cardano;
            }
        }
        rotate90(buffer, matrix_len);
    }

end_cardano:
    memcpy(to, buffer, size);
    return 0;
}

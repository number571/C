#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define ALPHABET_SORTED "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

typedef unsigned int uint;
typedef unsigned char uchar;
typedef struct {
    uchar *alpha;
    uint size;
} atbash_t;

extern atbash_t *atbash_new(uchar *alpha, uint size);
extern void atbash_free(atbash_t *cipher);

extern uchar atbash_encrypt(atbash_t *cipher, uchar ch);
static int find_index(uchar array[], uint size, uchar ch);

int main(int argc, char *argv[]) {
    atbash_t *cipher;
    int len;

    cipher = atbash_new(ALPHABET_SORTED, strlen(ALPHABET_SORTED));

    for (int i = 1; i < argc; ++i) {
        len = strlen(argv[i]);
        for (int j = 0; j < len; ++j) {
            putchar(atbash_encrypt(cipher, argv[i][j]));
        }
    }
    putchar('\n');

    atbash_free(cipher);

    return 0;
}

extern atbash_t *atbash_new(uchar *alpha, uint size) {
    atbash_t *cipher;

    cipher = (atbash_t*)malloc(sizeof(atbash_t));

    cipher->alpha = (uchar*)malloc(sizeof(uchar)*size);
    cipher->size = size;

    memcpy(cipher->alpha, alpha, size);

    return cipher;
}

extern void atbash_free(atbash_t *cipher) {
    free(cipher->alpha);
    free(cipher);
}

extern uchar atbash_encrypt(atbash_t *cipher, uchar ch) {
    int index;
    index = find_index(cipher->alpha, cipher->size, ch);
    if (index == -1) {
        return ch;
    }
    index = cipher->size - (index % cipher->size);
    return cipher->alpha[index];
}

static int find_index(uchar array[], uint size, uchar ch) {
    for (int i = 0; i < size; ++i) {
        if (array[i] == ch) {
            return i;
        }
    }
    return -1;
}

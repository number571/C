#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#define ALPHABET_SORTED "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

typedef unsigned int uint;
typedef unsigned char uchar;
typedef struct {
    uchar *alpha;
    uint size;
    uint key;
} caesar_t;

extern caesar_t *caesar_new(uint key, uchar *alpha, uint size);
extern void caesar_free(caesar_t *cipher);

extern uchar caesar_encrypt(caesar_t *cipher, uchar ch);
extern uchar caesar_decrypt(caesar_t *cipher, uchar ch);
static int find_index(uchar array[], uint size, uchar ch);

int main(int argc, char *argv[]) {
    caesar_t *cipher;
    uchar ch;
    int mode;
    int len;

    if (argc < 4) {
        fprintf(stderr, "use example: ./caesar [E|D] key message\n");
        return 1;
    }

    if (strcmp(argv[1], "E") == 0) {
        mode = 1;
    } else if (strcmp(argv[1], "D") == 0) {
        mode = -1;
    } else {
        mode = 0;
    }

    if (mode == 0) {
        fprintf(stderr, "undefined encryption mode\n");
        return 2;
    }

    cipher = caesar_new(atoi(argv[2]), ALPHABET_SORTED, strlen(ALPHABET_SORTED));

    for (int i = 3; i < argc; ++i) {
        len = strlen(argv[i]);
        for (int j = 0; j < len; ++j) {
            switch (mode) {
                case 1:
                    ch = caesar_encrypt(cipher, argv[i][j]);
                break;
                case -1:
                    ch = caesar_decrypt(cipher, argv[i][j]);
                break;
            }
            putchar(ch);
        }
    }
    putchar('\n');

    caesar_free(cipher);

    return 0;
}

extern caesar_t *caesar_new(uint key, uchar *alpha, uint size) {
    caesar_t *cipher;

    cipher = (caesar_t*)malloc(sizeof(caesar_t));

    cipher->alpha = (uchar*)malloc(sizeof(uchar)*size);
    cipher->size = size;
    cipher->key = key % size;

    memcpy(cipher->alpha, alpha, size);

    return cipher;
}

extern void caesar_free(caesar_t *cipher) {
    free(cipher->alpha);
    free(cipher);
}

extern uchar caesar_encrypt(caesar_t *cipher, uchar ch) {
    int index;
    index = find_index(cipher->alpha, cipher->size, ch);
    if (index == -1) {
        return ch;
    }
    index = (index + cipher->key) % cipher->size;
    return cipher->alpha[index];
}

extern uchar caesar_decrypt(caesar_t *cipher, uchar ch) {
    int index;
    index = find_index(cipher->alpha, cipher->size, ch);
    if (index == -1) {
        return ch;
    }
    index = (cipher->size + index - cipher->key) % cipher->size;
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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define ALPHABET_SORTED "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

typedef unsigned int uint;
typedef unsigned char uchar;
typedef struct {
    uint counter;
    uchar *alpha;
    uint asize;
    uchar *key;
    uint ksize;
} vigenere_t;

extern vigenere_t *vigenere_new(uchar *key, uint ksize, uchar *alpha, uint asize);
extern void vigenere_free(vigenere_t *cipher);

extern uchar vigenere_encrypt(vigenere_t *cipher, uchar ch);
extern uchar vigenere_decrypt(vigenere_t *cipher, uchar ch);

static int find_index(uchar array[], uint size, uchar ch);

int main(int argc, char *argv[]) {
    vigenere_t *cipher;
    uchar ch;
    int mode;
    int len;

    if (argc < 4) {
        fprintf(stderr, "use example: ./vigenere [E|D] key message\n");
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

    cipher = vigenere_new(argv[2], strlen(argv[2]), ALPHABET_SORTED, strlen(ALPHABET_SORTED));

    for (int i = 3; i < argc; ++i) {
        len = strlen(argv[i]);
        for (int j = 0; j < len; ++j) {
            switch(mode) {
                case 1:
                    ch = vigenere_encrypt(cipher, argv[i][j]);
                break;
                case -1:
                    ch = vigenere_decrypt(cipher, argv[i][j]);
                break;
            }
            putchar(ch);
        }
    }
    putchar('\n');

    vigenere_free(cipher);

    return 0;
}

extern vigenere_t *vigenere_new(uchar *key, uint ksize, uchar *alpha, uint asize) {
    vigenere_t *cipher;

    cipher = (vigenere_t*)malloc(sizeof(vigenere_t));

    cipher->counter = 0;

    cipher->key = (uchar*)malloc(sizeof(uchar)*ksize);
    cipher->ksize = ksize;
    memcpy(cipher->key, key, ksize);

    cipher->alpha = (uchar*)malloc(sizeof(uchar)*asize);
    cipher->asize = asize;
    memcpy(cipher->alpha, alpha, asize);

    return cipher;
}

extern uchar vigenere_encrypt(vigenere_t *cipher, uchar ch) {
    int index_ch, index_key;

    index_ch = find_index(cipher->alpha, cipher->asize, ch);
    if (index_ch == -1) {
        return ch;
    }

    index_key = find_index(cipher->alpha, cipher->asize, cipher->key[cipher->counter]);
    if (index_key == -1) {
        return ch;
    }

    index_ch = (index_ch + index_key) % cipher->asize;
    cipher->counter = (cipher->counter + 1) % cipher->ksize;

    return cipher->alpha[index_ch];
}

extern uchar vigenere_decrypt(vigenere_t *cipher, uchar ch) {
    int index_ch, index_key;

    index_ch = find_index(cipher->alpha, cipher->asize, ch);
    if (index_ch == -1) {
        return ch;
    }

    index_key = find_index(cipher->alpha, cipher->asize, cipher->key[cipher->counter]);
    if (index_key == -1) {
        return ch;
    }

    index_ch = (cipher->asize + index_ch - index_key) % cipher->asize;
    cipher->counter = (cipher->ksize + cipher->counter + 1) % cipher->ksize;

    return cipher->alpha[index_ch];
}

extern void vigenere_free(vigenere_t *cipher) {
    free(cipher->key);
    free(cipher->alpha);
    free(cipher);
}

static int find_index(uchar array[], uint size, uchar ch) {
    for (int i = 0; i < size; ++i) {
        if (array[i] == ch) {
            return i;
        }
    }
    return -1;
}

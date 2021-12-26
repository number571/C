#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define ALPHABET_SORTED "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

typedef unsigned int uint;
typedef unsigned char uchar;
typedef struct {
    uchar *alpha;
    uint size;
} bacon_t;

extern bacon_t *bacon_new(uchar *alpha, uint size);
extern void bacon_free(bacon_t *cipher);

extern uchar bacon_encrypt(bacon_t *cipher, uchar ch);
extern uchar bacon_decrypt(bacon_t *cipher, uchar ch);

static uchar bacon_decode(uchar *str);
static void bacon_print(uchar ch);

static int find_index(uchar array[], uint size, uchar ch);

int main(int argc, char *argv[]) {
    bacon_t *cipher;
    uchar ch;
    int mode;
    int len;

    if (argc < 3) {
        fprintf(stderr, "use example: ./bacon [E|D] message\n");
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

    cipher = bacon_new(ALPHABET_SORTED, strlen(ALPHABET_SORTED));

    for (int i = 2; i < argc; ++i) {
        len = strlen(argv[i]);
        switch (mode) {
            case 1:
                for (int j = 0; j < len; ++j) {
                    ch = bacon_encrypt(cipher, argv[i][j]);
                    bacon_print(ch);
                }
            break;
            case -1:
                ch = bacon_decrypt(cipher, bacon_decode(argv[i]));
                putchar(ch);
            break;
        }
    }
    putchar('\n');

    bacon_free(cipher);

    return 0;
}

extern bacon_t *bacon_new(uchar *alpha, uint size) {
    bacon_t *cipher;

    cipher = (bacon_t*)malloc(sizeof(bacon_t));

    cipher->alpha = (uchar*)malloc(sizeof(uchar)*size);
    cipher->size = size;

    memcpy(cipher->alpha, alpha, size);

    return cipher;
}

extern void bacon_free(bacon_t *cipher) {
    free(cipher->alpha);
    free(cipher);
}

extern uchar bacon_encrypt(bacon_t *cipher, uchar ch) {
    int index;
    index = find_index(cipher->alpha, cipher->size, ch);
    if (index == -1) {
        return -1;
    }
    return index;
}

extern uchar bacon_decrypt(bacon_t *cipher, uchar ch) {
    if (ch >= cipher->size) {
        return ch;
    }
    return cipher->alpha[ch];
}

static uchar bacon_decode(uchar *str) {
    uchar ch;
    ch = 0;
    for (int i = 0; i <= 7; ++i) {
        ch |= ((str[i] == 'B') ? 1 : 0) << (7-i);
    }
    return ch;
}

static void bacon_print(uchar ch) {
    for (int i = (1 << 7); i >= 1; i >>= 1) {
        putchar((ch & i) ? 'B' : 'A');
    }
    putchar(' ');
}

static int find_index(uchar array[], uint size, uchar ch) {
    for (int i = 0; i < size; ++i) {
        if (array[i] == ch) {
            return i;
        }
    }
    return -1;
}

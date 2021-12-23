#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define ALPHABET_SIZE   26
#define ALPHABET_SORTED "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

typedef unsigned int uint;
typedef struct {
    char *alpha;
    uint size;
} atbash_t;

extern atbash_t *atbash_new(char *alpha, uint size);
extern void atbash_free(atbash_t *cipher);

extern char atbash_encrypt(atbash_t *cipher, char ch);
static int find_index(char array[], int size, char ch);

int main(int argc, char *argv[]) {
    atbash_t *cipher;
    int len;

    cipher = atbash_new(ALPHABET_SORTED, ALPHABET_SIZE);

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

extern atbash_t *atbash_new(char *alpha, uint size) {
    atbash_t *cipher;

    cipher = (atbash_t*)malloc(sizeof(atbash_t));

    cipher->alpha = (char*)malloc(sizeof(char)*size);
    cipher->size = size;

    memcpy(cipher->alpha, alpha, size);

    return cipher;
}

extern void atbash_free(atbash_t *cipher) {
    free(cipher->alpha);
    free(cipher);
}

extern char atbash_encrypt(atbash_t *cipher, char ch) {
    int index;
    index = find_index(cipher->alpha, cipher->size, ch);
    if (index == -1) {
        return ch;
    }
    index = cipher->size - (index % cipher->size);
    return cipher->alpha[index];
}

static int find_index(char array[], int size, char ch) {
    for (int i = 0; i < size; ++i) {
        if (array[i] == ch) {
            return i;
        }
    }
    return -1;
}


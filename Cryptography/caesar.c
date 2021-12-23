#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#define CAESAR_KEY      3
#define ALPHABET_SIZE   26
#define ALPHABET_SORTED "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

typedef unsigned int uint;
typedef struct {
    char *alpha;
    uint size;
    uint key;
} caesar_t;

extern caesar_t *caesar_new(uint key, char *alpha, uint size);
extern void caesar_free(caesar_t *cipher);

extern char caesar_encrypt(caesar_t *cipher, char ch);
extern char caesar_decrypt(caesar_t *cipher, char ch);
static int find_index(char array[], int size, char ch);

int main(int argc, char *argv[]) {
    caesar_t *cipher;
    uint encch;
    int len;

    cipher = caesar_new(CAESAR_KEY, ALPHABET_SORTED, ALPHABET_SIZE);

    for (int i = 1; i < argc; ++i) {
        len = strlen(argv[i]);
        for (int j = 0; j < len; ++j) {
            putchar(caesar_encrypt(cipher, argv[i][j]));
        }
    }
    putchar('\n');

    caesar_free(cipher);

    return 0;
}

extern caesar_t *caesar_new(uint key, char *alpha, uint size) {
    caesar_t *cipher;

    cipher = (caesar_t*)malloc(sizeof(caesar_t));

    cipher->alpha = (char*)malloc(sizeof(char)*size);
    cipher->size = size;
    cipher->key = key % size;

    memcpy(cipher->alpha, alpha, size);

    return cipher;
}

extern void caesar_free(caesar_t *cipher) {
    free(cipher->alpha);
    free(cipher);
}

extern char caesar_encrypt(caesar_t *cipher, char ch) {
    int index;
    index = find_index(cipher->alpha, cipher->size, ch);
    if (index == -1) {
        return ch;
    }
    index = (index + cipher->key) % cipher->size;
    return cipher->alpha[index];
}

extern char caesar_decrypt(caesar_t *cipher, char ch) {
    int index;
    index = find_index(cipher->alpha, cipher->size, ch);
    if (index == -1) {
        return ch;
    }
    index = (cipher->size + index - cipher->key) % cipher->size;
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

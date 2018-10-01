#include "macro.h"
#include "types.h"

#include <string.h>

static char __alpha_vigenere[MAX_CHAR_QUANTITY] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static char __length_vigenere = LEN_ALPHA;

static char _char_vigenere (const char mode, char key, const char ch) {
    char c, *p = NULL;
    bool flag[2] = {false, false};

    for (p = __alpha_vigenere; *p != END_OF_STRING; ++p) {
        if (*p == ch) {
            c = (p - __alpha_vigenere);
            flag[0] = true;
        }
        if (*p == key) {
            key = (p - __alpha_vigenere) * mode;
            flag[1] = true;
        }
        if (flag[0] && flag[1])
            break;
    }

    if (flag[0] && flag[1])
        return __alpha_vigenere[(c + key + __length_vigenere) % __length_vigenere];

    return ch;
}

extern char set_alpha_vigenere (char * const alpha) {
    const unsigned int length = strlen(alpha);
    if (length >= MAX_CHAR_QUANTITY)
        return 1;

    __length_vigenere = (char)length;
    char *p = NULL;

    for (p = alpha; *p != END_OF_STRING; ++p)
        __alpha_vigenere[p - alpha] = *p;

    __alpha_vigenere[p - alpha] = END_OF_STRING;

    return 0;
}

extern char vigenere (
    char * const to,
    const char mode,
    const char * const key,
    char * const from
) {
    if (mode != ENCRYPT_MODE && mode != DECRYPT_MODE)
        return 1;

    const unsigned int length = strlen(key);
    char *p = NULL;

    for (p = from; *p != END_OF_STRING; ++p)
        to[p - from] = _char_vigenere(mode, key[(p - from) % length], *p);

    to[p - from] = END_OF_STRING;

    return 0;
}
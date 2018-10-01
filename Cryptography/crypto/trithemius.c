#include "macro.h"
#include <string.h>

static char __alpha_trithemius[MAX_CHAR_QUANTITY] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static char __length_trithemius = LEN_ALPHA;

static char _char_trithemius (const char mode, char key, const char ch) {
    char *p = NULL;
    for (p = __alpha_trithemius; *p != END_OF_STRING; ++p)
        if (*p == ch) {
            key = ( (key < 0) ? (__length_trithemius + (key % __length_trithemius)) : (key % __length_trithemius) ) * mode;
            return __alpha_trithemius[(p - __alpha_trithemius + key + __length_trithemius) % __length_trithemius];
        }
    return ch;
}

extern char set_alpha_trithemius (char * const alpha) {
    const unsigned int length = strlen(alpha);
    if (length >= MAX_CHAR_QUANTITY)
        return 1;

    __length_trithemius = (char)length;
    char *p = NULL;

    for (p = alpha; *p != END_OF_STRING; ++p)
        __alpha_trithemius[p - alpha] = *p;

    __alpha_trithemius[p - alpha] = END_OF_STRING;

    return 0;
}

extern char trithemius (
    char * const to, 
    const char mode, 
    char (*key) (const char x), 
    char * const from
) {
    if (mode != ENCRYPT_MODE && mode != DECRYPT_MODE)
        return 1;

    char *p = NULL;
    for (p = from; *p != END_OF_STRING; ++p)
        to[p - from] = \
            _char_trithemius(mode, key((char) (p - from)), *p);
            
    to[p - from] = END_OF_STRING;

    return 0;
}
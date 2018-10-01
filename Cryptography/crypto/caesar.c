#include "macro.h"
#include <string.h>

static char __alpha_caesar[MAX_CHAR_QUANTITY] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static char __length_caesar = LEN_ALPHA;

static char _char_caesar (const char key, const char ch) {
    char *p = NULL;
    for (p = __alpha_caesar; *p != END_OF_STRING; ++p)
        if (*p == ch)
            return __alpha_caesar[(p - __alpha_caesar + key + __length_caesar) % __length_caesar];
    return ch;
}

extern char set_alpha_caesar (char * const alpha) {
    const unsigned int length = strlen(alpha);
    if (length >= MAX_CHAR_QUANTITY)
        return 1;

    __length_caesar = (char)length;
    char *p = NULL;

    for (p = alpha; *p != END_OF_STRING; ++p)
        __alpha_caesar[p - alpha] = *p;

    __alpha_caesar[p - alpha] = END_OF_STRING;

    return 0;
}

extern char caesar (
    char * const to, 
    const char mode, 
    char key, 
    char * const from
) {
    if (mode != ENCRYPT_MODE && mode != DECRYPT_MODE)
        return 1;

    char *p = NULL;
    key = ( (key < 0) ? (__length_caesar + (key % __length_caesar)) : (key % __length_caesar) ) * mode;

    for (p = from; *p != END_OF_STRING; ++p)
        to[p - from] = _char_caesar(key, *p);
    
    to[p - from] = END_OF_STRING;

    return 0;
}
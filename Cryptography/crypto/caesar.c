#include <string.h>

#include "macro.h"

static char __alpha_caesar[MAX_LENGTH] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static unsigned char __length_caesar = LEN_ALPHA;

static char _char_caesar (const char key, const char ch) {
    for (char *p = __alpha_caesar; *p != END_OF_STRING; ++p)
        if (*p == ch)
            return __alpha_caesar[(p - __alpha_caesar + key + __length_caesar) % __length_caesar];
    return ch;
}

extern char set_alpha_caesar (const char * const alpha) {
    const size_t length = strlen(alpha);
    if (length >= MAX_LENGTH)
        return 1;

    __length_caesar = (unsigned char)length;

    strcpy(__alpha_caesar, alpha);
    return 0;
}

extern char caesar (
    char * to, 
    const char mode, 
    char key, 
    const char * from
) {
    if (mode != ENCRYPT_MODE && mode != DECRYPT_MODE)
        return 1;

    key = ( (key < 0) ? (__length_caesar + (key % __length_caesar)) : (key % __length_caesar) ) * mode;

    while (*from != END_OF_STRING)
        *to++ = _char_caesar(key, *from++);
    
    *to = END_OF_STRING;

    return 0;
}
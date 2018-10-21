#include <string.h>

#include "macro.h"

static char __alpha_atbash[MAX_LENGTH] = "ZYXWVUTSRQPONMLKJIHGFEDCBA";
static unsigned char __length_atbash = LEN_ALPHA;

static char _char_atbash (const char ch) {
    for (char *p = __alpha_atbash; *p != END_OF_STRING; ++p)
        if (*p == ch)
            return __alpha_atbash[__length_atbash - 1 - (p - __alpha_atbash)];
    return ch;
}

extern char set_alpha_atbash (const char * const alpha) {
    const size_t length = strlen(alpha);

    if (length >= MAX_LENGTH)
        return 1;

    __length_atbash = (unsigned char)length;
    strcpy(__alpha_atbash, alpha);

    return 0;
}

extern void atbash (char * to, const char * from) {
    while (*from != END_OF_STRING)
        *to++ = _char_atbash(*from++);
    
    *to = END_OF_STRING;
}
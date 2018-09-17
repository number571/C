#include <string.h>
#include "expansion.h"

char __alpha_atbash[MAX_CHAR_QUANTITY] = "ZYXWVUTSRQPONMLKJIHGFEDCBA";
char __length_atbash = LEN_ALPHA;

char _char_atbash (const char ch) {
    char *p = NULL;
    for (p = __alpha_atbash; *p != END_OF_STRING; ++p)
        if (*p == ch)
            return __alpha_atbash[__length_atbash - 1 - (p - __alpha_atbash)];
    return ch;
}

_Bool set_alpha_atbash (char * const alpha) {
    char length = strlen(alpha);
    if (length >= MAX_CHAR_QUANTITY)
        return 1;

    __length_atbash = length;
    char *p = __alpha_atbash;

    while (--length > 0)
        *p++ = alpha[length];
    *++p = END_OF_STRING;

    return 0;
}

void atbash (char * const to, char * const from) {
    char *p = NULL;
    for (p = from; *p != END_OF_STRING; ++p)
        *(to + (p - from)) = _char_atbash(*p);
    *(to + (p - from)) = END_OF_STRING;
}

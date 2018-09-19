#include <string.h>
#include "expansion.h"

char __alpha_substitute[MAX_CHAR_QUANTITY] = \
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*()_+=-`{}:\"<>?[];',./\\| ";

char __vector_substitute[MAX_CHAR_QUANTITY] = \
    "~!@#$%^&*()_+=-0123456789`{}:\"<>?[];',./\\| ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

_Bool set_alpha_substitute (char * const vector) {
    if (strlen(vector) >= MAX_CHAR_QUANTITY)
        return 1;

    char *p = NULL;
    for (p = vector; *p != END_OF_STRING; ++p)
        __alpha_substitute[p - vector] = *p;

    return 0;
}

_Bool set_vector_substitute (char * const vector) {
    if (strlen(vector) >= MAX_CHAR_QUANTITY)
        return 1;

    char *p = NULL;
    for (p = vector; *p != END_OF_STRING; ++p)
        __vector_substitute[p - vector] = *p;

    return 0;
}

char _char_substitute (char * const to, const char ch, char * const from) {
    char *p = NULL;
    for (p = from; *p != END_OF_STRING; ++p)
        if (ch == *p)
            return to[p - from];
    return ch;
}

_Bool substitute (
    char * const to, 
    const char mode, 
    char * const from
) {
    if (mode != ENCRYPT_MODE && mode != DECRYPT_MODE)
        return 1;

    char * const to_vector = (mode == ENCRYPT_MODE) ? __vector_substitute : __alpha_substitute;
    char * const from_vector = (mode == DECRYPT_MODE) ? __vector_substitute : __alpha_substitute;

    char *p = NULL;
    for (p = from; *p != END_OF_STRING; ++p)
        to [p - from] = _char_substitute(to_vector, *p, from_vector);
    to [p - from] = END_OF_STRING;

    return 0;
}
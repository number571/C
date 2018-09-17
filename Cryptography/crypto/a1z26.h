#include "expansion.h"

_Bool a1z26 (char * const to, const char mode, char * const from) {
    if (mode != ENCRYPT_MODE && mode != DECRYPT_MODE)
        return 1;

    char buffer[2];
    char *p = NULL;

    switch (mode) {
        case ENCRYPT_MODE:
            buffer[0] = 'A';
            buffer[1] = 'Z';
        break;
        case DECRYPT_MODE:
            buffer[0] = 1;
            buffer[1] = 26;
        break;
    }

    for (p = from; *p != END_OF_STRING; ++p)
        if (buffer[0] <= *p && *p <= buffer[1])
            to[p - from] = *p + (-mode * 'A') + mode;

    to[p - from] = END_OF_STRING;

    return 0;
}

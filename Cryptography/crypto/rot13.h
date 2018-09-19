#include "expansion.h"

char _char_rot13 (const char ch) {
    if ('A' <= ch && ch <= 'Z') 
        return ch % LEN_ALPHA + 'A';
    return ch;
}

void rot13 (char * const to, char * const from) {
    char *p = NULL;
    for (p = from; *p != END_OF_STRING; ++p)
        *(to + (p - from)) = _char_rot13(*p);
    *(to + (p - from)) = END_OF_STRING;
}
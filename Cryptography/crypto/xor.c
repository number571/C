#include "macro.h"
#define NULL ((void*) 0)

extern void xor (char * const to, const char key, char * const from) {
    char *p = NULL;
    for (p = from; *p != END_OF_STRING; ++p)
        to[p - from] = *p ^ key;
    
    to[p - from] = END_OF_STRING;
}
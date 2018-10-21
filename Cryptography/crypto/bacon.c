#include <stdio.h>
#include <string.h>

#include "macro.h"
#include "types.h"

static char __alpha_bacon[MAX_LENGTH] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static char __default_char_bacon[2] = {'A', 'B'};

static char _index_char_bacon (const char ch) {
    for (char *p = __alpha_bacon; *p != END_OF_STRING; ++p)
        if (*p == ch)
            return p - __alpha_bacon;
    return ch + END_OF_NUMBER;
}

static void _encrypt_bacon (char * to, const char * from) {
    for (; *from != END_OF_STRING; ++from)
        *to++ = _index_char_bacon(*from);

    *to = END_OF_NUMBER;
}

static void _decrypt_bacon (char * to, const char * from) {
    for (; *from != END_OF_NUMBER; ++from)
        if (*from < 0)
            *to++ = *from - END_OF_NUMBER;
        else
            *to++ = __alpha_bacon[*from];

    *to = END_OF_STRING;
}

extern void set_char_bacon (const char ch1, const char ch2) {
    __default_char_bacon[0] = ch1;
    __default_char_bacon[1] = ch2;
}

extern char set_alpha_bacon (const char * const alpha) {
    if (strlen(alpha) >= MAX_LENGTH)
        return 1;

    strcpy(__alpha_bacon, alpha);
    return 0;
}

extern void print_bacon (const char * from) {
    for (; *from != END_OF_NUMBER; ++from)
        if (*from >= 0) {
            Byte x = {.byte = *from};
            printf("%c%c%c%c%c%c%c%c", 
                x.bit._7 ? __default_char_bacon[1] : __default_char_bacon[0],
                x.bit._6 ? __default_char_bacon[1] : __default_char_bacon[0],
                x.bit._5 ? __default_char_bacon[1] : __default_char_bacon[0],
                x.bit._4 ? __default_char_bacon[1] : __default_char_bacon[0],
                x.bit._3 ? __default_char_bacon[1] : __default_char_bacon[0],
                x.bit._2 ? __default_char_bacon[1] : __default_char_bacon[0],
                x.bit._1 ? __default_char_bacon[1] : __default_char_bacon[0],
                x.bit._0 ? __default_char_bacon[1] : __default_char_bacon[0]);
        } else putchar(*from - END_OF_NUMBER);
}

extern char bacon (char * to, const char mode, const char * from) {
    switch (mode) {
        case ENCRYPT_MODE:
            _encrypt_bacon(to, from); 
        break;
        case DECRYPT_MODE:
            _decrypt_bacon(to, from); 
        break;
        default: return 1;
    }
    return 0;
}
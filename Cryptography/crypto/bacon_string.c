#include <math.h>
#include <string.h>

#include "macro.h"

#define SEVEN_BITS 7

static char __alpha_bacon_string[MAX_LENGTH] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static char __default_char_bacon_string[2] = {'A', 'B'};

static char _index_char_bacon_string (const char ch) {
    for (char *p = __alpha_bacon_string; *p != END_OF_STRING; ++p)
        if (*p == ch)
            return p - __alpha_bacon_string;
    return -1;
}

static void _encrypt_bacon_string (char * to, const char * const from) {
    const unsigned int length = strlen(from);

    char buffer[length + 1];
    strcpy(buffer, from);

    for (char *p = buffer; *p != END_OF_STRING; ++p) {
        const char index = _index_char_bacon_string(*p);

        if (index == -1) {
            *to++ = *p;
            continue;
        }
        
        for (char x = 0x40; x >= 0x01; x /= 2)
            *to++ = index & x ? __default_char_bacon_string[1] : __default_char_bacon_string[0];
    }

    *to = END_OF_STRING;
}

static char _from_vector_to_alpha_bacon_string (
    char * to,
    const char position,
    const char * const buffer
) {
    if (position != 0 && position % SEVEN_BITS == 0) {
    	char sum = 0;

        for (char x = SEVEN_BITS - 1; x > 0; --x)
            if (buffer[x] == __default_char_bacon_string[1])
                sum += (char)pow(2, SEVEN_BITS - 1 - x);

        *to++ = __alpha_bacon_string[sum];
        return 1;
    } 

    return 0;
}

static void _decrypt_bacon_string (char * to, const char * from) {
    char position = 0;
    char buffer[SEVEN_BITS + 1];

    for (; *from != END_OF_STRING; ++from) {
        to = _from_vector_to_alpha_bacon_string(to, position, buffer) ? (to + 1) : (to);
        position %= SEVEN_BITS;

        if (*from != __default_char_bacon_string[0] && *from != __default_char_bacon_string[1]) {
            *to++ = *from;
            continue;
        }

        buffer[position++] = *from;
    }

    to = _from_vector_to_alpha_bacon_string(to, position, buffer) ? (to + 1) : (to);
    position %= SEVEN_BITS;

    *to = END_OF_STRING;
}

extern void set_char_bacon_string (const char ch1, const char ch2) {
    __default_char_bacon_string[0] = ch1;
    __default_char_bacon_string[1] = ch2;
}

extern char set_alpha_bacon_string (const char * const alpha) {
    if (strlen(alpha) >= MAX_LENGTH)
        return 1;

    strcpy(__alpha_bacon_string, alpha);
    return 0;
}

extern char bacon_string (char * to, const char mode, const char * from) {
    switch (mode) {
        case ENCRYPT_MODE:
            _encrypt_bacon_string(to, from); 
        break;
        case DECRYPT_MODE:
            _decrypt_bacon_string(to, from); 
        break;
        default: return 1;
    }
    return 0;
}
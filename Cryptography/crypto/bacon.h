#include <math.h>
#include <string.h>
#include "expansion.h"

char __alpha_bacon[MAX_CHAR_QUANTITY] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
char __vector_bacon[2] = {'A', 'B'};

char _char_bacon (const char ch, const char x) {
    char *p = NULL;
    for (p = __alpha_bacon; *p != END_OF_STRING; ++p)
        if (*p == ch)
            return (p - __alpha_bacon) & x ? __vector_bacon[1] : __vector_bacon[0];
    return ch;
}

void set_vector_bacon (const char first, const char second) {
    __vector_bacon[0] = first;
    __vector_bacon[1] = second;
}

_Bool set_alpha_bacon (char * const alpha) {
    if (strlen(alpha) >= MAX_CHAR_QUANTITY)
        return 1;

    char *p = NULL;
    for (p = alpha; *p != END_OF_STRING; ++p)
        __alpha_bacon[p - alpha] = *p;

    __alpha_bacon[p - alpha] = END_OF_STRING;

    return 0;
}

void _encrypt_bacon (char * const to, char * const from) {
    unsigned int position = 0;
    char x, *p = NULL;
    
    for (p = from; *p != END_OF_STRING; ++p)
        for (x = 0x40; x >= 0x01; x /= 2) {
            to[position++] = _char_bacon(*p, x);
            if (to[position - 1] != __vector_bacon[0] && to[position - 1] != __vector_bacon[1])
                break;
        }

    to[position] = END_OF_STRING;
}

unsigned int _from_vector_to_alpha_bacon (
    char * const to,
    unsigned int position,
    char * const position_buffer,
    const char * const buffer
) {
    char x, sum = 0;

    if (*position_buffer != 0 && *position_buffer % SEVEN_BITS == 0) {
        for (x = SEVEN_BITS - 1; x > 0; --x)
            if (buffer[x] == __vector_bacon[1])
                sum += pow(2, SEVEN_BITS - 1 - x);

        to[position++] = __alpha_bacon[sum];
        sum = *position_buffer = 0;
    } 

    return position;
}

void _decrypt_bacon (char * const to, char * const from) {
    unsigned int position = 0;
    char position_buffer = 0;

    char x, *p = NULL;
    char buffer[SEVEN_BITS + 1];

    for (p = from; *p != END_OF_STRING; ++p) {
        position = _from_vector_to_alpha_bacon(to, position, &position_buffer, buffer);

        if (*p != __vector_bacon[0] && *p != __vector_bacon[1]) {
            to[position++] = *p;
            continue;
        }

        buffer[position_buffer++] = *p;
    }

    position = _from_vector_to_alpha_bacon(to, position, &position_buffer, buffer);

    to[position] = END_OF_STRING;
}

_Bool bacon (char * const to, const char mode, char * const from) {
    switch(mode) {
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
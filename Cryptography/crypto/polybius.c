#include "macro.h"
#include <string.h>

static char __alpha_polybius[MAX_CHAR_QUANTITY] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static char __length_polybius = LEN_ALPHA;

static char _get_length_of_square_polybius (void) {
    char square;
    for (square = 1; (square * square) < __length_polybius; ++square)
        ;
    return square;
}

static void _push_square_to_buffer_polybius (char * const buffer, const char square) {
    unsigned int position = 0;
    char x, y;
    for (x = 10; x <= square * 10; x += 10)
        for (y = 1; y <= square; ++y)
            buffer[position++] = x + y;

    buffer[position] = END_OF_STRING;
}

static char _char_polybius (
    const char ch, 
    const char mode,
    const char * const first, 
    const char * const second
) {
    char x;
    for (x = 0; x < __length_polybius; ++x)
        if (ch == first[x])
            return second[x];
    return (mode == ENCRYPT_MODE) ? (ch - 128) : (ch + 128);
}

extern char set_alpha_polybius (char * const alpha) {
    const unsigned int length = strlen(alpha);
    if (length >= MAX_CHAR_QUANTITY)
        return 1;

    __length_polybius = (char)length;
    char *p = NULL;

    for (p = alpha; *p != END_OF_STRING; ++p)
        __alpha_polybius[p - alpha] = *p;
    
    __alpha_polybius[p - alpha] = END_OF_STRING;

    return 0;
}

extern char polybius (char * const to, const char mode, char * const from) {
    if (mode != ENCRYPT_MODE && mode != DECRYPT_MODE)
        return 1;    

    const char square = _get_length_of_square_polybius();
    
    char buffer[square * square + 1];
    _push_square_to_buffer_polybius(buffer, square);

    char *p = NULL;
    const char * const p_first  = (mode == ENCRYPT_MODE) ? __alpha_polybius : buffer;
    const char * const p_second = (mode == ENCRYPT_MODE) ? buffer : __alpha_polybius;

    for (p = from; *p != END_OF_STRING; ++p)
        to[p - from] = _char_polybius(*p, mode, p_first, p_second);

    to[p - from] = END_OF_STRING;

    return 0;
}
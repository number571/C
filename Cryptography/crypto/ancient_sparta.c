#include "macro.h"
#include <string.h>

static char __default_char_ancient_sparta = 'Z';

static void _encrypt_ancient_sparta (
    char * const to, const char key, char * const from
) {
    unsigned int x, start_length, final_length;
    unsigned int y, block_length, position = 0;

    start_length = final_length = strlen(from);

    while (final_length % key != 0)
        ++final_length;
    block_length = final_length / key;

    char buffer[final_length + 1];
    strcpy(buffer, from);

    while (start_length++ < final_length)
        buffer[start_length-1] = __default_char_ancient_sparta;

    for (x = 0; x < block_length; ++x)
        for (y = x; y < final_length; y += block_length)
            to[position++] = buffer[y];

    to[position] = END_OF_STRING;
}

static void _decrypt_ancient_sparta (
    char * const to, const char key, char * const from
) {
    const unsigned int length = strlen(from);
    unsigned int x, y, position = 0;

    char buffer[length + 1];

    for (x = 0; x < key; ++x)
        for (y = x; y < length; y += key)
            buffer[position++] = from[y];

    buffer[position] = END_OF_STRING;
    strcpy(to, buffer);
}

extern void set_char_ancient_sparta (const char ch) {
    __default_char_ancient_sparta = ch;
}

extern char ancient_sparta (
    char * const to, 
    const char mode, 
    const char key, 
    char * const from
) {
    if (key < 1) return 1;
    switch (mode) {
        case ENCRYPT_MODE: 
            _encrypt_ancient_sparta(to, key, from); 
        break;
        case DECRYPT_MODE: 
            _decrypt_ancient_sparta(to, key, from); 
        break;
        default: return 2;
    }
    return 0;
}
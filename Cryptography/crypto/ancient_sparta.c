#include <string.h>

#include "macro.h"

static char __default_char_ancient_sparta = 'Z';

static void _encrypt_ancient_sparta (
    char * to, const char key, const char * const from
) {
    size_t start_length, final_length;
    start_length = final_length = strlen(from);

    while (final_length % key != 0)
        ++final_length;

    const size_t block_length = final_length / key;

    char buffer[final_length + 1];
    strcpy(buffer, from);

    while (start_length++ < final_length)
        buffer[start_length - 1] = __default_char_ancient_sparta;

    for (size_t x = 0; x < block_length; ++x)
        for (size_t y = x; y < final_length; y += block_length)
            *to++ = buffer[y];

    *to = END_OF_STRING;
}

static void _decrypt_ancient_sparta (
    char * const to, const char key, const char * const from
) {
    const size_t length = strlen(from);

    char buffer[length + 1];
    char *p = buffer;

    for (size_t x = 0; x < key; ++x)
        for (size_t y = x; y < length; y += key)
            *p++ = from[y];

    *p = END_OF_STRING;

    strcpy(to, buffer);
}

extern void set_char_ancient_sparta (const char ch) {
    __default_char_ancient_sparta = ch;
}

extern char ancient_sparta (
    char * to, 
    const char mode, 
    const char key, 
    const char * const from
) {
    if (key < 1) return 2;

    switch (mode) {
        case ENCRYPT_MODE: 
            _encrypt_ancient_sparta(to, key, from); 
        break;
        case DECRYPT_MODE: 
            _decrypt_ancient_sparta(to, key, from); 
        break;
        default: return 1;
    }
    return 0;
}
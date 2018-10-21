#include "macro.h"
#include "types.h"

#include <string.h>

static char __alpha_porte[MAX_CHAR_QUANTITY] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static char __length_porte = LEN_ALPHA;
static char __default_char_porte = 'Z';

static unsigned int _get_length (short * const from, const char end) {
    short *p = from;
    while (*++p != end);
    return p - from;
}

static void _encrypt_porte (short * const to, short * const from) {
    Point number = {0, 0};
    bool flag[2] = {false, false};

    char i;

    short *p = NULL;
    short *p_to = to;

    const unsigned int length = _get_length(from, END_OF_STRING);
    short buffer[length + 2];

    copy(short, buffer, from, END_OF_STRING);

    if (length % 2 != 0) {
        buffer[length] = __default_char_porte;
        buffer[length + 1] = END_OF_STRING;
    }
    
    for (p = buffer; *p != END_OF_STRING; p += 2) {
        for (i = 0; i < __length_porte; ++i) {
            if (*p == __alpha_porte[i]) {
                number.x = i;
                flag[0] = true;
            }

            if (*(p + 1) == __alpha_porte[i]) {
                number.y = i;
                flag[1] = true;
            }

            if (flag[0] && flag[1])
                break;
        }
        *p_to++ = (number.x * __length_porte) + number.y;
        flag[0] = flag[1] = false;
    }

    *p_to = END_OF_NUMBER;
}

static void _decrypt_porte (short * const to, short * const from) {
    const unsigned int length = _get_length(from, END_OF_NUMBER) * 2;
    short buffer[length + 1];

    unsigned int position = 0;
    short *p = NULL;

    for (p = from; *p != END_OF_NUMBER; ++p) {
        buffer[position++] = __alpha_porte[*p / __length_porte];
        buffer[position++] = __alpha_porte[*p % __length_porte];
    }

    buffer[position] = END_OF_STRING;
    copy(short, to, buffer, END_OF_STRING);
}

extern void set_char_porte (const char ch) {
    __default_char_porte = ch;
}

extern char set_alpha_porte (char * const alpha) {
    const unsigned int length = strlen(alpha);
    if (length >= MAX_CHAR_QUANTITY)
        return 1;

    __length_porte = (char)length;
    copy(char, __alpha_porte, alpha, END_OF_STRING);

    return 0;
}

extern char porte (
    short * const to,
    const char mode,
    short * const from
) {
    switch (mode) {
        case ENCRYPT_MODE:
            _encrypt_porte(to, from);
        break;
        case DECRYPT_MODE:
            _decrypt_porte(to, from);
        break;
        default: return 1;
    }
    
    return 0;
}
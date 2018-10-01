#include "macro.h"
#include "types.h"

#include <string.h>

#define SUBST(x) ((x == 0) ? (*position) : (length))
#define SQUARE(x) (x * x)

static char __alpha_playfair[MAX_CHAR_QUANTITY] = "ABCDEFGHIKLMNOPQRSTUVWXYZ";
static char __length_playfair = LEN_ALPHA - 1;
static char __default_char_playfair = 'X';

static bool _get_coordinates_playfair (
    Point * place,
    const unsigned int num,
    const char length
) {
    place->x = num / length;
    place->y = num % length;
    return true;
}

static char _get_length_of_square_playfair (void) {
    char length;
    for (length = 1; SQUARE(length) != __length_playfair; ++length)
        ;
    return length;
}

static void _push_to_buffer_playfair (
    const char mode,
    char * const buffer, 
    char * position,
    char * const key
) {
    _Bool flag;

    char x, *p = NULL;
    char length = *position;

    for (p = key; *p != END_OF_STRING; ++p) {
        flag = 0;
        for (x = 0; x < SUBST(mode); ++x)
            if (*p == buffer[x]) {
                flag = 1;
                break;
            }
        if (!flag)
            buffer[(*position)++] = *p;
    }
}

static void _push_key_to_buffer_playfair (
    char * const buffer, 
    char * const key
) {
     char position = 0;

    _push_to_buffer_playfair(0, buffer, &position, key);
    _push_to_buffer_playfair(1, buffer, &position, __alpha_playfair);

    buffer[position] = END_OF_STRING;
}

static void _shift_playfair (
    const char shift_mode,
    char * const ch,
    const char mode,
    const char * const key,
    const char length,
    const Point place[2]
) {
    char z[2], i;
    for (i = 0; i < 2; ++i) {
        z[i] = (!shift_mode) ? (place[i].y) : (place[i].x);
        z[i] += mode;

        z[i] = (z[i] >= length) ? (z[i] % length) : (z[i]);
        z[i] = (z[i] < i) ? ((z[i] + length) % length) : (z[i]);

        *(ch + i) = (!shift_mode) ? \
            ( key[(place[i].x * length) + z[i]] ) : \
            ( key[(z[i] * length) + place[i].y] );
    }
}

static void _chars_playfair (
    char * const ch,
    const char mode,
    char * const key,
    const char length
) {
    Point place[2];
    
    bool flag[2] = {false, false};
    char *p = NULL;

    for (p = key; *p != END_OF_STRING && !(flag[0] && flag[1]); ++p) {
        if (*p == *ch)
            flag[0] = _get_coordinates_playfair(&place[0], (p - key), length);

        if (*p == *(ch + 1))
            flag[1] = _get_coordinates_playfair(&place[1], (p - key), length);
    }

    if (place[0].x == place[1].x)
        _shift_playfair(0, ch, mode, key, length, place);

    else if (place[0].y == place[1].y)
        _shift_playfair(1, ch, mode, key, length, place);

    else {
        *ch = key[(place[1].x * length) + place[0].y];
        *(ch + 1) = key[(place[0].x * length) + place[1].y];
    }
}

static char _change_the_message (
    char * const to, 
    char * const from
) {
    unsigned int position = 0;
    char *p = NULL;

    for (p = from; *p != END_OF_STRING && *(p + 1) != END_OF_STRING; p += 2) {
        if (*p == *(p + 1)) {
            if (*p == __default_char_playfair && *(p + 1) == __default_char_playfair)
                return 1;

            to[position++] = *p;
            to[position++] = __default_char_playfair;
            to[position++] = *p;
        } else {
            to[position++] = *p;
            to[position++] = *(p + 1);
        }
    }

    if (*p != END_OF_STRING)
        to[position++] = from[p - from];

    to[position] = END_OF_STRING;

    if (strlen(to) % 2 != 0) {
        to[position++] = __default_char_playfair;
        to[position] = END_OF_STRING;
    }

    return 0;
}

extern void set_char_playfair (const char ch) {
    __default_char_playfair = ch;
}

extern char set_alpha_playfair (char * const alpha) {
    const unsigned int length = strlen(alpha);
    if (length >= MAX_CHAR_QUANTITY)
        return 1;

    char length_of_square;
    for (length_of_square = 1; SQUARE(length_of_square) < MAX_CHAR_QUANTITY; ++length_of_square)
        if (SQUARE(length_of_square) == length)
            break;

    if (SQUARE(length_of_square) >= MAX_CHAR_QUANTITY)
        return 2;

    char x, *p = NULL;
    for (p = alpha; *p != END_OF_STRING; ++p)
        for (x = p - alpha + 1; x < length; ++x)
            if (*p == alpha[x])
                return 3;

    __length_playfair = (char)length;

    for (p = alpha; *p != END_OF_STRING; ++p)
        __alpha_playfair[p - alpha] = *p;
    
    __alpha_playfair[p - alpha] = END_OF_STRING;

    return 0;
}

extern char playfair (
    char * const to, 
    const char mode, 
    char * const key,
    char * const from
) {
    if (mode != ENCRYPT_MODE && mode != DECRYPT_MODE)
        return 1;

    if (_change_the_message(to, from))
        return 2;

    char *p = NULL;
    const char length = _get_length_of_square_playfair();

    char buffer_key[SQUARE(length) + 1];
    _push_key_to_buffer_playfair(buffer_key, key);

    for (p = to; *p != END_OF_STRING; p += 2)
        _chars_playfair(p, mode, buffer_key, length);

    to[p - from] = END_OF_STRING;

    return 0;
}
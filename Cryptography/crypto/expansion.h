#pragma once

#define END_OF_STRING '\0'
#define LEN_ALPHA 26

#define ENCRYPT_MODE  1
#define DECRYPT_MODE -1

#define SEVEN_BITS 7
#define MAX_CHAR_QUANTITY 100

struct coordinates { char x; char y; };

_Bool _get_coordinates (
    struct coordinates * place,
    const unsigned int num,
    const char length
) {
    place->x = num / length;
    place->y = num % length;
    return 1;
}

void print_nums (char * const array) {
    char *p = NULL;
    for (p = array; *p != 0; ++p)
        printf("%hhd ", *p);
    printf("\n");
}

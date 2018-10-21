#pragma once

#define ENCRYPT_MODE  1
#define DECRYPT_MODE -1

#define END_OF_STRING  0
#define END_OF_NUMBER -128

#define LEN_ALPHA 26
#define MAX_LENGTH 127

#define \
    copy(Type, to, from, end) \
    ({ \
        Type *p = from; \
        while (*p != end) { \
            to[p - from] = *p; \
            ++p; \
        } \
        to[p - from] = end; \
    })

#define \
    print_nums(Type, from) \
    ({ \
        for (Type *p = from; *p != END_OF_NUMBER; ++p) \
            printf("%d ", *p); \
        printf("\n"); \
    })

#define \
    print_chars(Type, from) \
    ({ \
        for (Type *p = from; *p != END_OF_STRING; ++p) \
            printf("%c", *p); \
        printf("\n"); \
    })

#define \
    print_byte(from) \
    ({ \
        unsigned char *p = &from; \
        Byte x = {.byte = *p}; \
        printf("%hhu%hhu%hhu%hhu%hhu%hhu%hhu%hhu", \
                x.bit._7, x.bit._6, x.bit._5, x.bit._4, \
                x.bit._3, x.bit._2, x.bit._1, x.bit._0); \
    }) \

#define \
    print_bytes(from) \
    ({ \
        for (char *p = from; *p != END_OF_NUMBER; ++p) { \
            Byte x = {.byte = *p}; \
            printf("%hhu%hhu%hhu%hhu%hhu%hhu%hhu%hhu", \
                x.bit._7, x.bit._6, x.bit._5, x.bit._4, \
                x.bit._3, x.bit._2, x.bit._1, x.bit._0); \
        } \
    })

#include <string.h>

#include "macro.h"
#include "types.h"

static char __alpha_one_couple[MAX_LENGTH] = "ACEGIKMOQSUWY";
static char __alpha_two_couple[MAX_LENGTH] = "BDFHJLNPRTVXZ";

static unsigned char __length_couple = LEN_ALPHA / 2;

extern char set_alpha_couple (char * const alph1, char * const alph2) {
    const size_t length = strlen(alph1);

    if (length >= MAX_LENGTH)
        return 1;

    if (length != strlen(alph2))
        return 2;

    for (char *px = alph1; *px != END_OF_STRING; ++px)
        for (char *py = alph2; *py != END_OF_STRING; ++py)
            if (*px == *py)
                return 3;

    __length_couple = (unsigned char)length;
    unsigned char i = 0;

    while (i < __length_couple) {
        __alpha_one_couple[i] = alph1[i];
        __alpha_two_couple[i] = alph2[i];
        ++i;
    }

    __alpha_one_couple[i] = __alpha_two_couple[i] = END_OF_STRING;

    return 0;
}

extern void couple (char * to, const char * from) {
    for (bool flag; *from != END_OF_STRING; ++from, flag = false) {
        for (unsigned char x = 0; x < __length_couple; ++x) {

            if (*from == __alpha_one_couple[x]) {
                *to++ = __alpha_two_couple[x];
                flag = true;
                break;

            } else if (*from == __alpha_two_couple[x]) {
                *to++ = __alpha_one_couple[x];
                flag = true;
                break;
            }
        }

        if (!flag) *to++ = *from;
    }  
    
    *to = END_OF_STRING;
}
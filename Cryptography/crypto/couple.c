#include "macro.h"
#include "types.h"

#include <string.h>

static char __alpha_couple[MAX_CHAR_QUANTITY]  = "ACEGIKMOQSUWY";
static char __vector_couple[MAX_CHAR_QUANTITY] = "BDFHJLNPRTVXZ";

extern char set_alpha_couple (char * const vector) {
    if (strlen(vector) >= MAX_CHAR_QUANTITY)
        return 1;

    char *p = NULL;
    for (p = vector; *p != END_OF_STRING; ++p)
        __alpha_couple[p - vector] = *p;

    return 0;
}

extern char set_vector_couple (char * const vector) {
    if (strlen(vector) >= MAX_CHAR_QUANTITY)
        return 1;

    char *p = NULL;
    for (p = vector; *p != END_OF_STRING; ++p)
        __vector_couple[p - vector] = *p;

    return 0;
}

extern char couple (char * const to, char * const from) {
    const char length_alpha = strlen(__alpha_couple);

    if (length_alpha != strlen(__vector_couple))
        return 1;

    bool flag;
    char x, *p = NULL;

    for (p = from; *p != END_OF_STRING; ++p) {
        flag = false;

        for (x = 0; x < length_alpha; ++x) {
            if (*p == __alpha_couple[x]) {
                to[p - from] = __vector_couple[x];
                flag = true;
                break;
            }

            else if (*p == __vector_couple[x]) {
                to[p - from] = __alpha_couple[x];
                flag = true;
                break;
            }
        }

        if (!flag) 
            to[p - from] = *p;
    }  

    to[p - from] = END_OF_STRING;

    return 0;
}
#include <stdio.h>
#define END_OF_STRING '\0'

void atbash (char * const to, char * const from);
char char_atbash (const char ch);

int main(void) {
    char message[] = "HELLO, WORLD";

    atbash(message, message);
    printf("%s\n", message);

    return 0;
}

void atbash (char * const to, char * const from) {
    char *p = NULL;
    for (p = from; *p != END_OF_STRING; ++p)
        *(to + (p - from)) = char_atbash(*p);
    *(to + (p - from)) = END_OF_STRING;
}

char char_atbash (const char ch) {
    char *reversed = "ZYXWVUTSRQPONMLKJIHGFEDCBA";
    if ('A' <= ch && ch <= 'Z') 
        return reversed[ch - 'A'];
    return ch;
}

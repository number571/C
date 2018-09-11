#include <stdio.h>

void atbash (char to[], char from[]);
char char_atbash (const char ch);

int main(void) {
    char message[] = "HELLO, WORLD";
    atbash(message, message);
    printf("%s\n", message);
    return 0;
}

void atbash (char to[], char from[]) {
    char *p = NULL;
    for (p = from; *p != '\0'; ++p)
        *(to + (p - from)) = char_atbash(*p);
    *(to + (p - from)) = '\0';
}

char char_atbash (const char ch) {
    char *reversed = "ZYXWVUTSRQPONMLKJIHGFEDCBA";
    if ('A' <= ch && ch <= 'Z') 
        return reversed[ch - 'A'];
    return ch;
}

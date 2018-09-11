#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *atbash (char *message);
char char_atbash (char ch);

int main(void) {
    printf("%s\n", atbash("HELLO, WORLD"));
    return 0;
}

char *atbash (char *message) {
    unsigned long length = strlen(message);
    unsigned long i;

    char *encrypted_message = NULL;
    encrypted_message = (char*)malloc(length + 1);

    for (i = 0; i < length; ++i)
        encrypted_message[i] = char_atbash(message[i]);
    encrypted_message[i] = '\0';

    return encrypted_message;
}

char char_atbash (char ch) {
    char *reversed = "ZYXWVUTSRQPONMLKJIHGFEDCBA";

    if ('A' <= ch && ch <= 'Z') 
        return reversed[ch - 'A'];

    else return ch;
}

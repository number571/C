#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LEN_ALPHA 26

#define ENCRYPT  1
#define DECRYPT -1

char *caesar (char mode, char key, char *message);
char encrypt_char (char key, char ch);

int main (void) {

    printf("%s\n", caesar(ENCRYPT, 3, "HELLO, WORLD"));

    return 0;
}

char *caesar (char mode, char key, char *message) {
    unsigned long length = strlen(message);
    unsigned long i;

    key = ( (key < 0) ? (LEN_ALPHA + key) : (key) ) * mode;

    char *encrypted_message = NULL;
    encrypted_message = (char*)malloc(length + 1);

    for (i = 0; i < length; ++i)
        encrypted_message[i] = encrypt_char(key, message[i]);

    return encrypted_message;
}

char encrypt_char (char key, char ch) {
    if ('A' <= ch && ch <= 'Z') 
        return (ch + key + 13) % LEN_ALPHA + 'A';

    else return ch;
}

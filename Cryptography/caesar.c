#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define BUFFER 128

void encryptDecrypt(char mode, char *message, short key);

int main(void) {
    char mode; short key;
    scanf("%c %hd ", &mode, &key);

    mode = toupper(mode);
    char text[BUFFER], temp;
    for (unsigned short index = 0; index < BUFFER; index++)
        if ((temp = getchar()) != '\n')
            text[index] = temp;
        else break;

    for (unsigned short index = 0; index < strlen(text); index++)
        text[index] = toupper(text[index]);

    encryptDecrypt(mode, text, key);
    printf("Final message: %s\n", text);
    return 0;
}

void encryptDecrypt(char mode, char *message, short key) {
    if (mode == 'E')
        for (unsigned short index = 0; index < strlen(message); index++)
            message[index] = (message[index] + key - 13) % 26 + 'A';
    else
        for (unsigned short index = 0; index < strlen(message); index++)
            message[index] = (message[index] - key - 13) % 26 + 'A';
}

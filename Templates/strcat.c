#include <stdio.h>

#define LIMIT 128

void strcat_ (char str[], char t[]);

int main (void) {
    char string[LIMIT] = "hello, ";
    strcat_(string, "world");
    printf("%s\n", string);
    return 0;
}

void strcat_ (char str[], char t[]) {
    int i, j;
    i = j = 0;
    while (str[i] != '\0')
        i++;
    while ((str[i++] = t[j++]) != '\0');
}

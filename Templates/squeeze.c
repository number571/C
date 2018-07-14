#include <stdio.h>

typedef enum {false, true} bool;

void squeeze (char str[], char rmv[]);

int main (void) {
    char string[] = "hello, world";
    char remove[] = "how";

    squeeze(string, remove);

    printf("%s\n", string);
    return 0;
}

void squeeze (char str[], char rmv[]) {
    bool flag;
    int i, j, c;
    for (i = j = 0; str[i] != '\0'; i++) {
        flag = true;
        for (c = 0; rmv[c] != '\0'; c++)
            if (str[i] == rmv[c]) {
                flag = false;
                break;
            } 
        if (flag)
            str[j++] = str[i];
    }
    str[j] = '\0';
}

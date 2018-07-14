#include <stdio.h>

#define LIMIT 128

int getline_ (char str[], int lim);

int main (void) {
    char string[LIMIT];
    int quan = getline_(string, LIMIT);
    printf("%d - %s\n", quan, string);
    return 0;
}

int getline_ (char str[], int lim) {
    int c, i = 0;
    while (i < lim-1 && (c = getchar()) != EOF && c != '\n')
        str[i++] = c;
    if (c == '\n')
        str[i++] = c;
    str[i] = '\0';
    return i;
}

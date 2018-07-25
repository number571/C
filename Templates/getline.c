#include <stdio.h>

#define LIMIT 128

int getline_ (char s[], int lim);

int main (void) {
    char string[LIMIT];
    int quan = getline_(string, LIMIT);
    printf("%d - %s\n", quan, string);
    return 0;
}

int getline_ (char s[], int lim) {
    int c, i = 0;
    while (--lim > 0 && (c = getchar()) != EOF && c != '\n')
        s[i++] = c;
    if (c == '\n')
        s[i++] = c;
    s[i] = '\0';
    return i;
}

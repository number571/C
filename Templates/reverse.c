#include <stdio.h>

int strlen_ (char str[]);
void reverse (char s[]);

int main (void) {
    char string[] = "hello, world";
    reverse(string);
    printf("%s\n", string);
    return 0;
}

void reverse (char s[]) {
    int c, i, j;
    for (i = 0, j = strlen_(s)-1; i < j; i++, j--)
        c = s[i], s[i] = s[j], s[j] = c;
}

int strlen_ (char str[]) {
    int i = 0;
    while (str[i++] != '\0');
    return i-1;
}

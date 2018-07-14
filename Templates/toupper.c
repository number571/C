#include <stdio.h>

int toupper_ (char c);

int main (void) {
    char symbol = 'r';
    printf("%c\n", toupper_(symbol));
    return 0;
}

int toupper_ (char c) {
    if (c >= 'a' && c <= 'z')
        return c + 'A' - 'a';
    else return c;
}

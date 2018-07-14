#include <stdio.h>

int tolower_ (char c);

int main (void) {
    char symbol = 'R';
    printf("%c\n", tolower_(symbol));
    return 0;
}

int tolower_ (char c) {
    if (c >= 'A' && c <= 'Z')
        return c + 'a' - 'A';
    else return c;
}

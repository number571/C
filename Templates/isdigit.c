#include <stdio.h>

int isdigit_ (char c);

int main (void) {
    char symbol = '5';
    printf("%d\n", isdigit_(symbol));
    return 0;
}

int isdigit_ (char c) {
    if (c >= '0' && c <= '9')
        return 1;
    else return 0;
}

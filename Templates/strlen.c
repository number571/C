#include <stdio.h>

int strlen_ (char str[]);

int main (void) {
    char string[] = "hello, world";
    printf("%d\n", strlen_(string));
    return 0;
}

int strlen_ (char str[]) {
    int i = 0;
    while (str[i++] != '\0');
    return i-1;
}

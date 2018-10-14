#include <stdio.h>

void getLine (char * const s, const size_t max);

int main (void) {
    char string[100];
    getLine(string, 100);
    printf("%s\n", string);
    return 0;
}

void getLine (char * const s, const size_t max) {
    char *p = s;
    while ((*p = getchar()) != '\n' && p - s < max - 1)
        ++p;
    *p = '\0';
}

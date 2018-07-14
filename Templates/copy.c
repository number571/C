#include <stdio.h>

#define LIMIT 128

void copy (char to[], char from[]);

int main (void) {
    char string[LIMIT] = "hello, world";
    char copy_string[LIMIT];

    copy(copy_string, string);

    printf("%s\n", copy_string);
    return 0;
}

void copy (char to[], char from[]) {
    int i = 0;
    while ((to[i] = from[i]) != '\0')
        ++i;
}

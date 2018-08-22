#include <stdio.h>
int main(int argc, char const *argv[]) {
    ++argv;
    while (--argc > 0)
        printf((argc == 1)?"%s\n":"%s ", *argv++);
    return 0;
}
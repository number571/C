#include <stdio.h>

int isspace_ (char c);

int main (void) {
    printf("%d\n", isspace_(' '));
    return 0;
}

int isspace_ (char c) {
    switch (c) {
        case ' ': case '\n': case '\t': return 1;
        default: return 0;
    }
}

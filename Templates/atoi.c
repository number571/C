#include <stdio.h>

int atoi (char str[]);

int main (void) {
    char num_string[] = "123";
    printf("%d\n", atoi(num_string));
    return 0;
}

int atoi (char str[]) {
    int i, n;
    i = n = 0;
    while (str[i] >= '0' && str[i] <= '9')
        n = 10 * n + (str[i++] - '0');
    return n;
}

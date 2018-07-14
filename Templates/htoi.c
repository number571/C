#include <stdio.h>

int htoi (char str[]);

int main (void) {
    char num_string[] = "0x16";
    printf("%d\n", htoi(num_string));
    return 0;
}

int htoi (char str[]) {
    int i, n;
    i = n = 0;
    if (str[0] == '0' && (str[1] == 'x' || str[i] == 'X'))
        i = 2;
    while (str[i] >= '0' && str[i] <= '9')
        n = 16 * n + (str[i++] - '0');
    return n;
}

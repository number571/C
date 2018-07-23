#include <stdio.h>

void itoa (int n, char s[]);
void reverse (char s[]);
int strlen_ (char str[]);

int main (void) {
    char string[20];
    itoa(571, string);
    printf("%s\n", string);
    return 0;
}

void itoa (int n, char s[]) {
    int i, sign;
    if ((sign = n) < 0)
        n = -n;
    i = 0;
    do {
        s[i++] = n % 10 + '0';
    } while ((n /= 10) > 0);
    if (sign < 0)
        s[i++] = '-';
    s[i] = '\0';
    reverse(s);
}

void reverse (char s[]) {
    int c, i, j;
    for (i = 0, j = strlen_(s)-1; i < j; i++, j--) {
        c = s[i]; 
        s[i] = s[j]; 
        s[j] = c;
    }
}

int strlen_ (char str[]) {
    int i = 0;
    while (str[i++] != '\0');
    return i-1;
}

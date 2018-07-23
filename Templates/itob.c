#include <stdio.h>

void itob (int n, char s[], int b);
void reverse (char s[]);
int strlen_ (char str[]);

int main (void) {
    char string[20];
    itob(15, string, 16);
    printf("%s\n", string);
    return 0;
}

void itob (int n, char s[], int b) {
    int i, sign, c;
    if ((sign = n) < 0)
        n = -n;
    i = 0;
    do {
        c = (n % b);
        s[i++] = (c>9) ? c+'A'-10: c + '0';
    } while ((n /= b) >= 1);
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

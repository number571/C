#include <stdio.h>

int isdigit_ (char c);
int isspace_ (char c);
int atoi_ (char s[]);

int main (void) {
    printf("%d\n", atoi_("-571"));
    return 0;
}

int atoi_ (char s[]) {
    int i, n, sign;
    for (i = 0; isspace_(s[i]); i++);
    sign = (s[i] == '-') ? -1 : 1;
    if (s[i] == '+' || s[i] == '-')
        i++;
    for (n = 0; isdigit_(s[i]); i++)
        n = 10 * n + (s[i] - '0');
    return sign * n;
}

int isspace_ (char c) {
    switch (c) {
        case ' ': case '\n': case '\t': return 1;
        default: return 0;
    }
}

int isdigit_ (char c) {
    if (c >= '0' && c <= '9')
        return 1;
    else return 0;
}

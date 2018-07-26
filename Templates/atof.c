#include <stdio.h>

int isdigit_ (char c);
int isspace_ (char c);
double atof (char s[]);

int main (void) {
    printf("%lf\n", atof("57.1"));
    return 0;
}

double atof (char s[]) {
    double val, power;
    int i, sign;
    for (i = 0; isspace_(s[i]); i++);
    sign = (s[i] == '-') ? -1 : 1;
    if (s[i] == '+' || s[i] == '-')
        i++;
    for (val = 0.0; isdigit_(s[i]); i++)
        val = 10.0 * val + (s[i] - '0');
    if (s[i] == '.')
        i++;
    for (power = 1.0; isdigit_(s[i]); i++) {
        val = 10.0 * val + (s[i] - '0');
        power *= 10;
    }
    return sign * val / power;
}

int isdigit_ (char c) {
    if (c >= '0' && c <= '9')
        return 1;
    else return 0;
}

int isspace_ (char c) {
    switch (c) {
        case ' ': case '\n': case '\t': return 1;
        default: return 0;
    }
}

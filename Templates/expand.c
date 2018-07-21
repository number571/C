#include <stdio.h>
#define SIZE 128

int strlen_ (char str[]);
void expand(char s1[], char s2[]);

int main (void) {
    char example[] = "a-z0-9";
    char string[SIZE];

    expand(example, string);
    printf("%s\n", string);
    return 0;
}

void expand(char s1[], char s2[]) {
    int i, j, c, len;
    len = strlen_(s1);
    j = 0;
    for (i = 0; i < len; i++)
        if (i + 2 < len)
            if (s1[i+1] == '-') {
                for (c = s1[i]; c <= s1[i+2]; c++)
                    s2[j++] = c;
                i += 2;
            } else s2[j++] = s1[i];
        else s2[j++] = s1[i];
    s2[j] = '\0';
}

int strlen_ (char str[]) {
    int i = 0;
    while (str[i++] != '\0');
    return i-1;
}

#include <stdio.h>

int strindex (char s[], char t[]);

int main (void) {
    char example[] = "or";
    char string[] = "hello, world";
    printf("%d\n", strindex(string, example));
    return 0;
}

int strindex (char s[], char t[]) {
    int i, j, k;
    for (i = 0; s[i] != '\0'; i++) {
        for (j = i, k = 0; t[k] != '\0' && s[j] == t[k]; j++, k++);
        if (k > 0 && t[k] == '\0')
            return i;
    }
    return -1;
}

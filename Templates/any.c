#include <stdio.h>

int any (char str[], char exm[]);

int main (void) {
    char string[] = "hello, world";
    char example[] = "world";

    printf("%d\n", any(string, example));
    return 0;
}

int any (char str[], char exm[]) {
    int i, j;
    for (i = 0; str[i] != '\0'; i++)
        for (j = 0; exm[j] != '\0'; j++)
            if (str[i] == exm[j])
                return i;
    return -1;
}

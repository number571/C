#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ENCRYPTION_MODE "-e"
#define DECRYPTION_MODE "-d"

#define LEN_ALPHA 26
#define LEN_DIGIT 10

#define ENCRYPT 1
#define DECRYPT 0

// ./main -e 3 hello, world

void encrypt_char (_Bool mode, char key, char ch);
_Bool check_args (int argc, char *argv[]);
void get_error (char *err);

int main(int argc, char *argv[]) {

    _Bool mode = check_args(argc, argv);

    unsigned short length;
    unsigned char i;

    char key = (char)(atoi(argv[2]) % LEN_ALPHA);
    key *= (key < 0) ? -1 : 1;
    printf("%hhd\n", key);

    char **p = (argv + 3);

    while (p < argv + argc) {
        length = strlen(*p);
        for (i = 0; i < length; ++i)
            encrypt_char(mode, key, (*p)[i]);
        putchar(' ');
        ++p;
    }

    printf("\n");

    return 0;
}

void encrypt_char (_Bool mode, char key, char ch) {
    key *= mode ? 1 : -1;

    if ('A' <= ch && ch <= 'Z') 
        putchar((ch + key + 13) % LEN_ALPHA + 'A');

    else if ('a' <= ch && ch <= 'z')
        putchar((ch + key + 7) % LEN_ALPHA + 'a');

    else if ('0' <= ch && ch <= '9')
        putchar((ch + key + 2) % LEN_DIGIT + '0');

    else putchar(ch);
}

_Bool check_args (int argc, char *argv[]) {
    if (argc < 4) get_error("len argc /= 4;");

    if (!strcmp(argv[1], ENCRYPTION_MODE))
        return ENCRYPT;

    else if (!strcmp(argv[1], DECRYPTION_MODE))
        return DECRYPT;

    else get_error("Mode is not '-e' / '-d';");
}

void get_error (char *err) {
    printf("Error: %s\n", err);
    exit(1);
}

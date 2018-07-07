#!/usr/bin/tcc -run
/* COMPILER: TCC */

#include <stdio.h>

#define OPEN 1
#define CLOSE 0

char checkSyntax(const char* const filename);

int main (void) {

    char err = checkSyntax("file.c");

    switch(err) {
        case  0: printf("SUCCESS: 'NO ERROR'\n"); break;
        case -1: printf("ERROR: 'File is not open'\n"); break;
        case  1: printf("ERROR: '() || {} || []'\n"); break;
        case  2: printf("ERROR: 'string or char is OPEN'\n"); break;
        case  3: printf("ERROR: '/* comments */'\n"); break;
        case  4: printf("ERROR: 'special symbols'\n");
    }

    return 0;
}

char checkSyntax(const char* const filename) {

    char comment, r_bracket, c_bracket, s_bracket;
    comment = r_bracket = c_bracket = s_bracket = 0;

    _Bool b_string, b_char;
    b_string = b_char = CLOSE;

    _Bool s_symbol = 0;

    FILE *file = fopen(filename, "r");

    if (file != NULL) {

        char c;
        unsigned int index = 1;

        while ((c = getc(file)) != EOF) {

            printf("%c", c);

            switch(c) {

                case '"':
                    if (b_string == CLOSE)
                        b_string = OPEN;
                    else b_string = CLOSE;
                    break;

                case '\'':
                    if (b_char == CLOSE)
                        b_char = OPEN;
                    else b_char = CLOSE;
                    break;

                case '(': ++r_bracket; break;
                case ')': --r_bracket; break;
                case '{': ++c_bracket; break;
                case '}': --c_bracket; break;
                case '[': ++s_bracket; break;
                case ']': --s_bracket; break;

                case '/':
                    if ((c = getc(file)) == '*')
                        ++comment; break;

                case '*':
                    if ((c = getc(file)) == '/')
                        --comment; break;

                case '\\':
                    c = getc(file);
                    if (c != 'r' && c != 'v' &&
                        c != 'b' && c != 'n' && 
                        c != 't' && c != 'e')
                        s_symbol = 1;
            }
            fseek(file, index++, SEEK_SET);
        }
        fclose(file);

    } else return -1;

    printf("\n");

    if (r_bracket != 0 || c_bracket != 0 || s_bracket != 0)
        return 1;

    if (b_string == OPEN || b_char == OPEN)
        return 2;

    if (comment != 0)
        return 3;

    if (s_symbol)
        return 4;

    return 0;
}

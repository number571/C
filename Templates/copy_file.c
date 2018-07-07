#include <stdio.h>
#define BUFF 256

_Bool copyFile(const char* const readfile, const char* const copyfile);

int main(void) {
    copyFile("file.txt", "hello.txt");
    return 0;
}

_Bool copyFile(const char* const readfile, const char* const copyfile) {
    char localBuffer[BUFF];
    FILE *read = fopen(readfile, "r"); 
    FILE *copy = fopen(copyfile, "w");
    if ((read != NULL) && (copy != NULL)) {
        while(fgets(localBuffer, BUFF, read) != NULL)
            fputs(localBuffer, copy);
    } else return 1;
    return 0;
}

#include "library.h"

#include <stdio.h>
#include <dirent.h>

extern char array_changes[QUAN];
extern unsigned char position;

static char check_exist_dir (const char* const dirname) {
    DIR* const dir = opendir(dirname);
    if (dir != NULL) {
        closedir(dir);
        return READABLE;
    } else return UNREADABLE;
}

static char check_exist_file (const char* const filename) {
    FILE* const file = fopen(filename, "r");
    if (file != NULL) {
        fclose(file);
        return READABLE;
    } else return UNREADABLE;
}

extern void check_main_dir (void) {
    const char exist_main_dir = check_exist_dir(MAIN_DIR);
    const char exist_hostname = check_exist_file(HOST_FILE);
    const char exist_private_key = check_exist_file(KEY_FILE);

    printf("[D_%s] => %s\n", CHECK_EXIST(exist_main_dir), MAIN_DIR);
    printf("[F_%s] => %s\n", CHECK_EXIST(exist_main_dir), HOST_FILE);
    printf("[F_%s] => %s\n", CHECK_EXIST(exist_main_dir), KEY_FILE);

    array_changes[position++] = exist_hostname;
    array_changes[position++] = exist_private_key;
}

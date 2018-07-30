#include "library.h"

#include <stdio.h>
#include <dirent.h>

extern char array_changes[QUAN];
extern unsigned char position;

extern void check_main_dir (void);
static char check_exist_dir (char *dirname);
static char check_exist_file (char *filename);

extern void check_main_dir (void) {
    auto char exist_main_dir = check_exist_dir(MAIN_DIR);
    auto char exist_hostname = check_exist_file(HOST_FILE);
    auto char exist_private_key = check_exist_file(KEY_FILE);

    printf("[D_%s] => %s\n", CHECK_EXIST(exist_main_dir), MAIN_DIR);
    printf("[F_%s] => %s\n", CHECK_EXIST(exist_main_dir), HOST_FILE);
    printf("[F_%s] => %s\n", CHECK_EXIST(exist_main_dir), KEY_FILE);

    array_changes[position++] = exist_hostname;
    array_changes[position++] = exist_private_key;
}

static char check_exist_dir (char *dirname) {
    auto DIR *dir = opendir(dirname);
    if (dir != NULL) {
        closedir(dir);
        return READABLE;
    } else return UNREADABLE;
}

static char check_exist_file (char *filename) {
    auto FILE *file = fopen(filename, "r");
    if (file != NULL) {
        fclose(file);
        return READABLE;
    } else return UNREADABLE;
}

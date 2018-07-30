#include "library.h"

#include <stdio.h>
#include <stdlib.h>

extern char array_changes[QUAN];
extern unsigned char position;

extern void edit_readme (void);
static char check_exist_file (const char* const filename);
static char* const read_file (const char* const filename);

extern void edit_readme (void) {
    auto struct List st_readme = { UNREADABLE, README_PATH };
    array_changes[position++] = check_exist_file(README_PATH);

    auto FILE *readme;
    auto bool changes = false;
    auto unsigned char index;

    for (index = 0; index < QUAN; index++)
        if (array_changes[index] != READABLE) {
            changes = true;
            break;
        }

    if (!changes) st_readme.mode = READABLE;

    else if ((readme = fopen(st_readme.path, "w")) != NULL) {
        fprintf( readme,
            "Tor configuration: %s:\n"
            "-   %s\n"
            "-   %s\n\n"
            "HTML file: %s\n"
            "Main files: %s\n\n"
            "-   hostname [%s]:\n%s\n"
            "-   private_key [%s]:\n%s\n"
            "Use this command for run tor service:\n"
            "[for one time, after rebooting use this command again]\n"
            "-   systemctl start tor.service\n"
            "[for always time]\n"
            "-   systemctl enable tor.service\n"
            "Use this command for activate port 80:\n"
            "[use in the directory: %s]\n"
            "-   python3 -m http.server 80\n",
        TORRC_PATH, HIDDEN_SERVICE_DIR, HIDDEN_SERVICE_PORT,
        HTML_FILE_PATH, MAIN_DIR, HOST_FILE, read_file(HOST_FILE), 
        KEY_FILE, read_file(KEY_FILE), ONION_PATH);

        st_readme.mode = OVERWRITTEN;
        fclose(readme);
    }

    printf("[F_%s] => %s\n", CHECK_MODE(st_readme.mode), st_readme.path);
}

static char check_exist_file (const char* const filename) {
    auto FILE* const file = fopen(filename, "r");
    if (file != NULL) {
        fclose(file);
        return READABLE;
    } else return UNREADABLE;
}

static char* const read_file (const char* const filename) {
    auto char c;
    auto unsigned int index = 0; 
    auto FILE* const file = fopen(filename, "r");
    if (file != NULL) {
        fseek(file, 0, SEEK_END);
            auto const unsigned int length = ftell(file);
        fseek(file, 0, SEEK_SET);

        auto char *content = (char*) malloc(length * sizeof(char));
        while((c = getc(file)) != EOF)
            *(content + index++) = c;
        *(content + index) = '\0';

        fclose(file);
        return content;
    } else return NULL;
}

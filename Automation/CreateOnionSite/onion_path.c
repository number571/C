#include "library.h"

#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>

static char* const get_html_code (void) {
    return
    "<!DOCTYPE html>\n"
    "<html>\n"
    "   <head>\n"
    "       <title>hello, world</title>\n"
    "       <meta charset='utf-8'>\n"
    "   </head>\n"
    "   <body>\n"
    "       <p>The_site_was_raised_from_#571</p>\n"
    "   </body>\n"
    "</html>\n";
}

static char edit_file (struct List *st_file) {
    FILE *file;
    if ((file = fopen(st_file->path, "r")) != NULL) {
        st_file->mode = READABLE;
        fclose(file);
    } else if ((file = fopen(st_file->path, "w")) != NULL) {
        auto char *pointer = get_html_code();
        st_file->mode = OVERWRITTEN;
        while (*pointer != '\0')
            putc(*pointer++, file);
        fclose(file);
    }
    return st_file->mode;
}

static char edit_dir (struct List *st_dir) {
    DIR *directory;
    if ((directory = opendir(st_dir->path)) != NULL) {
        st_dir->mode = READABLE;
        closedir(directory);
    } else if (!mkdir(st_dir->path, 0777)) {
        st_dir->mode = OVERWRITTEN;
    }
    return st_dir->mode;
}

extern void create_onion (void) {
    struct List www   = { UNREADABLE, WWW_PATH };
    struct List onion = { UNREADABLE, ONION_PATH };
    struct List html  = { UNREADABLE, HTML_FILE_PATH };

    edit_dir(&www); edit_dir(&onion); edit_file(&html);

    printf("[D_%s] => %s\n", CHECK_MODE(www.mode), www.path);
    printf("[D_%s] => %s\n", CHECK_MODE(onion.mode), onion.path);
    printf("[F_%s] => %s\n", CHECK_MODE(html.mode), html.path);

    if (html.mode == OVERWRITTEN)
        printf("| => Code HTML [ADDED] => %s\n", html.path);

    array_changes[position++] = www.mode;
    array_changes[position++] = onion.mode;
    array_changes[position++] = html.mode;
}

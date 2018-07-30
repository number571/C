#include "library.h"

#include <stdio.h>
#include <string.h>

extern char array_changes[QUAN];
extern unsigned char position;

extern void check_torrc (void);
static char edit_torrc  (struct List *st_torrc);

extern void check_torrc (void) {
    auto struct List torrc  = { UNREADABLE, TORRC_PATH };

    edit_torrc(&torrc); 
    printf("[F_%s] => %s\n", CHECK_MODE(torrc.mode), torrc.path);
    if (torrc.mode == OVERWRITTEN)
        printf("| => Lines [ADDED] => %s\n", torrc.path);

    array_changes[position++] = torrc.mode;
}

static char edit_torrc (struct List *st_torrc) {
    auto struct Data hsd = { false, HIDDEN_SERVICE_DIR  "\n"};
    auto struct Data hsp = { false, HIDDEN_SERVICE_PORT "\n"};

    auto char buffer[BUFF];
    auto FILE *torrc;

    if ((torrc = fopen(st_torrc->path, "r")) != NULL) {
        while (fgets(buffer, BUFF, torrc) != NULL) {
            if (!strcmp(buffer, hsd.string))
                hsd.exist = true;
            if (!strcmp(buffer, hsp.string))
                hsp.exist = true;
        }
        fclose(torrc);
    }

    if (hsd.exist && hsp.exist) {
        st_torrc->mode = READABLE;

    } else if ((torrc = fopen(st_torrc->path, "a")) != NULL) {
        if (!hsd.exist) fputs(hsd.string, torrc);
        if (!hsp.exist) fputs(hsp.string, torrc);
        st_torrc->mode = OVERWRITTEN;
        fclose(torrc);
    }

    return st_torrc->mode;
}

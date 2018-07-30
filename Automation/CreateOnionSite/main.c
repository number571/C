#include "library.h"

/* [Example]: $ make -> $ sudo ./main */

extern void create_onion (void);
extern void check_torrc (void);
extern void check_main_dir (void);
extern void edit_readme (void);

/* From sys_call.c */
extern void start_tor_service (void);
extern void run_server (void);

char array_changes[QUAN];
unsigned char position = 0;

int main (void) {

    create_onion();
    check_torrc();
    check_main_dir();
    start_tor_service();
    edit_readme();
    run_server();

    return 0;
}

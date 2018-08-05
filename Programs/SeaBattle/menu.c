#include "library.h"

#include <stdio.h>
#include <ctype.h>

#define MENU_X 16
#define MENU_Y 6

extern char getch (void);       /* FROM platform.c */
extern void clear (void);       /* FROM platform.c */

extern bool menu_mode;          /* TO main.c */
bool menu_mode = true;

extern void get_menu (void);    /* TO main.c */

static void print_menu (void);              /* IN menu.c */
static void change_menu_mode (bool mode);   /* IN menu.c */

static char menu[MENU_Y][MENU_X] = {
    " ============== ",
    " | START GAME | ",
    " ============== ",
    " -------------- ",
    " | STOP  GAME | ",
    " -------------- "
};

extern void get_menu (void) {
    auto char symbol;

    while(true) {
        print_menu();
        symbol = getch();

        switch (toupper(symbol)) {
            case 'A': case 'D': case 13: return;
            case 'W': change_menu_mode(true); break;
            case 'S': change_menu_mode(false); break;
            default: break;
        }
    }
}

static void print_menu (void) {
    auto unsigned char x, y;
    clear(); printf("\n\v");
    for (y = 0; y < MENU_Y; y++) {
        printf("\t");
        for (x = 0; x < MENU_X; x++)
            printf("%c", menu[y][x]);
        printf("\n");
    }
}

static void change_menu_mode (bool mode) {
    auto unsigned x;
    auto char c1, c2;

    c1 = mode?'=':'-';
    c2 = mode?'-':'=';

    for (x = 1; x < 15; x++) {
        menu[0][x] = c1;
        menu[2][x] = c1;
        menu[3][x] = c2;
        menu[5][x] = c2;
    }

    menu_mode = mode;
    print_menu();
}

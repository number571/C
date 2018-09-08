#include "library.h"

#include <stdio.h>
#include <ctype.h>


extern char getch (void);
extern void clear (void);
extern bool menu_mode;


static char menu[MENU_Y][MENU_X] = {
    " ============== ",
    " | START GAME | ",
    " ============== ",
    " -------------- ",
    " | STOP  GAME | ",
    " -------------- "
};

static void print_menu (void) {
    unsigned char x, y;
    clear(); printf("\n\v");
    for (y = 0; y < MENU_Y; y++) {
        printf("\t");
        for (x = 0; x < MENU_X; x++)
            printf("%c", menu[y][x]);
        printf("\n");
    }
}

static void change_menu_mode (bool mode) {
    unsigned x;
    char c1, c2;

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

extern void get_menu (void) {
    char symbol;

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

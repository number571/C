#include "library.h"

#include <stdio.h>

extern void print_map (void);   /* TO game.c && action.c */
static void print_result (const char* const s); /* IN main.c */

extern unsigned char size_ship;     /* TO game.c && action.c */
unsigned char size_ship = 4;

extern unsigned char count_ship;    /* TO game.c && action.c */
unsigned char count_ship = 10;

char map[MAP_Y][MAP_X] = {      /* TO game.c && action.c */
    "   A B C D E F G H I J    /     A B C D E F G H I J ",
    " 0| | | | | | | | | | |   /   0| | | | | | | | | | |",
    " 1| | | | | | | | | | |   /   1| | | | | | | | | | |",
    " 2| | | | | | | | | | |   /   2| | | | | | | | | | |",
    " 3| | | | | | | | | | |   /   3| | | | | | | | | | |",
    " 4| | | | | | | | | | |   /   4| | | | | | | | | | |",
    " 5| | | | | | | | | | |   /   5| | | | | | | | | | |",
    " 6| | | | | | | | | | |   /   6| | | | | | | | | | |",
    " 7| | | | | | | | | | |   /   7| | | | | | | | | | |",
    " 8| | | | | | | | | | |   /   8| | | | | | | | | | |",
    " 9| | | | | | | | | | |   /   9| | | | | | | | | | |"
};

extern char getch (void);           /* FROM platform.c */
extern void clear (void);           /* FROM platform.c */

extern void get_menu (void);        /* FROM menu.c */
extern bool menu_mode;              /* FROM menu.c */

extern void move_ships (void);      /* FROM action.c */
extern bool exit_mode;              /* FROM action.c */

extern void start_game (bool s);    /* FROM game.c */
extern char win;                    /* FROM game.c */

int main (void) {
    while (true) {

    get_menu();
    
    if (menu_mode) {
        move_ships();

        if (!exit_mode) {
            start_game(true);

            while (win == -1) {
                get_menu();
                if (!menu_mode)
                    goto end;
                start_game(false);
            }

            if (win == 1) print_result("WIN ");
            else if (win == 0) print_result("LOSE");
            break;

        } else continue;

    } else break;

    }

    end: 
    if (!menu_mode) clear();

    return 0;
}

static void print_result (const char* const s) {
    printf("\n\n"\
        "\n /*************************/"\
        "\n /*      RESULT: %s     */"\
        "\n /*************************/"\
    "\n\n\n", s);
}

extern void print_map (void) {
    auto unsigned char x, y;
    clear();
    for (y = 0; y < MAP_Y; y++) {
        for (x = 0; x < MAP_X; x++)
            printf("%c", map[y][x]);
        printf("\n");
    }
}

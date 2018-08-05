#include "library.h"

#include <stdlib.h>
#include <ctype.h>
#include <time.h>

#define TARGET '+'
#define HIT  'X' 
#define MISS '~'

#define OPP_MAP_X 52
#define OPP_MAP_Y 11

#define DEFAULT_X_OPP 32
#define DEFAULT_Y_OPP 1

extern unsigned char size_ship;     /* FROM main.c */
extern unsigned char count_ship;    /* FROM main.c */

extern char map[MAP_Y][MAP_X];      /* FROM main.c */
extern void print_map (void);       /* FROM main.c */

extern char getch (void);           /* FROM platform.c */

extern char win;                    /* TO main.c */
char win = -1;     

extern void start_game (bool continue_game);      /* TO main.c */

static void generate_opp_map (void);    /* IN game.c */
static bool boom_opp (void);            /* IN game.c */
static char boom (void);                /* IN game.c */
static void check_space (void);         /* IN game.c */
static void check_target (void);        /* IN game.c */
static void move_left_opp (void);       /* IN game.c */
static void move_right_opp (void);      /* IN game.c */
static void move_top_opp (void);        /* IN game.c */
static void move_bottom_opp (void);     /* IN game.c */

static unsigned char X_opp = DEFAULT_X_OPP;
static unsigned char Y_opp = DEFAULT_Y_OPP;

static unsigned char count_damaged_ship = 0;
static unsigned char count_damaged_opp_ship = 0;

static char map_opp[OPP_MAP_Y][OPP_MAP_X] = {
    "                          /     A B C D E F G H I J ",
    "                          /   0| | | | | | | | | | |",
    "                          /   1| | | | | | | | | | |",
    "                          /   2| | | | | | | | | | |",
    "                          /   3| | | | | | | | | | |",
    "                          /   4| | | | | | | | | | |",
    "                          /   5| | | | | | | | | | |",
    "                          /   6| | | | | | | | | | |",
    "                          /   7| | | | | | | | | | |",
    "                          /   8| | | | | | | | | | |",
    "                          /   9| | | | | | | | | | |"
};

extern void start_game (bool start) {
    if (start) {
        srand(time(NULL));

        size_ship = 4;
        count_ship = 10;

        generate_opp_map();
        map[Y_opp][X_opp] = TARGET;
    }

    auto char symbol;
    auto char result;

    while(true) {
        print_map();
        symbol = getch();

        switch (toupper(symbol)) {

            case 'A': move_left_opp(); break;
            case 'D': move_right_opp(); break;
            case 'W': move_top_opp(); break;
            case 'S': move_bottom_opp(); break;

            case 13 : 
                result = boom();
                
                if (result == 1) { win = 1; return; } 
                else if (result == -1) { win = 0; return; } 
                else break;

            case 27: return;
            default: break;
        }
    }
}

static void generate_opp_map (void) {
    auto unsigned char x, y;
    auto unsigned char counter;
    auto bool vertical;

    again:
        vertical = rand() % 2;
        x = 32 + rand() % 20;
        x = (x % 2 == 0)?(x):(x - 1);
        y = rand() % 10 + 1;

    if (vertical) {
        if (y + size_ship > 10)
            goto again;

        for (counter = 0; counter < size_ship; counter++)
            if (map_opp[y+counter][x] == SET_BLOCK)
                goto again;

        for (counter = 0; counter < size_ship; counter++)
            map_opp[y+counter][x] = SET_BLOCK;

    } else {
        if (x + size_ship > 49)
            goto again;

        for (counter = 0; counter < size_ship; counter++)
            if (map_opp[y][x+counter*2] == SET_BLOCK)
                goto again;

        for (counter = 0; counter < size_ship; counter++)
            map_opp[y][x+counter*2] = SET_BLOCK;
    }

    if (count_ship != 1) {
        --count_ship;
        switch(count_ship) {
            case 9: size_ship = 3; break;
            case 7: size_ship = 2; break;
            case 4: size_ship = 1; break;
            default: break;
        }
        goto again;
    } else return;
}

static bool boom_opp (void) {
    auto unsigned char x, y;

    again:
        x = 3 + rand() % 20;
        x = (x % 2 != 0)?(x):(x - 1);
        y = rand() % 10 + 1;

    if (map[y][x] == MISS || map[y][x] == HIT) 
        goto again;

    if (map[y][x] == SET_BLOCK) {

        map[y][x] = HIT;
        ++count_damaged_ship;

        if (count_damaged_ship == 20) {
            print_map();
            return true;
        }
    } else if (map[y][x] == SPACE) {
        map[y][x] = MISS;
    }

    return false;
}

static char boom (void) {
    if (map_opp[Y_opp][X_opp] == SET_BLOCK &&
        map_opp[Y_opp][X_opp] != HIT) {

        map[Y_opp][X_opp] = HIT;
        map_opp[Y_opp][X_opp] = HIT;

        ++count_damaged_opp_ship;
        if (count_damaged_opp_ship == 20) {
            print_map();
            return 1;
        }
        if (boom_opp()) return -1;

    } else if (map_opp[Y_opp][X_opp] == SPACE){

        map[Y_opp][X_opp] = MISS;
        map_opp[Y_opp][X_opp] = MISS;

        if (boom_opp()) return -1;
    }
    return 0;
}

static void check_space (void) {
    if (map[Y_opp][X_opp] == HIT)
        map[Y_opp][X_opp] = HIT;

    else if (map[Y_opp][X_opp] == MISS)
        map[Y_opp][X_opp] = MISS;

    else map[Y_opp][X_opp] = SPACE;
}

static void check_target (void) {
    if (map[Y_opp][X_opp] == HIT)
        map[Y_opp][X_opp] = HIT;

    else if (map[Y_opp][X_opp] == MISS)
        map[Y_opp][X_opp] = MISS;

    else map[Y_opp][X_opp] = TARGET;
}

static void move_left_opp (void) {
    check_space();
    if (X_opp != 32) X_opp -= 2;
    check_target();
}

static void move_right_opp (void) {
    check_space();
    if (X_opp != 50) X_opp += 2;
    check_target();
}

static void move_top_opp (void) {
    check_space();
    if (Y_opp != 1) --Y_opp;
    check_target();
}

static void move_bottom_opp (void) {
    check_space();
    if (Y_opp != 10) ++Y_opp;
    check_target();
}

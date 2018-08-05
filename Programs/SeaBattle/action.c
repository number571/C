#include "library.h"

#include <ctype.h>

#define BLOCK '@'

#define DEFAULT_X 3
#define DEFAULT_Y 1

extern unsigned char size_ship;     /* FROM main.c */
extern unsigned char count_ship;    /* FROM main.c */

extern char map[MAP_Y][MAP_X];      /* FROM main.c */
extern void print_map (void);       /* FROM main.c */

extern char getch (void);           /* FROM platform.c */

extern bool exit_mode;              /* TO main.c */
bool exit_mode = false;

extern void move_ships (void);      /* TO main.c */

static void generate_ship (void);   /* IN action.c */
static void reverse_ship (void);    /* IN action.c */
static void move_left (void);       /* IN action.c */
static void move_right (void);      /* IN action.c */
static void move_top (void);        /* IN action.c */
static void move_bottom (void);     /* IN action.c */
static bool set_ship (void);        /* IN action.c */

static unsigned char X = DEFAULT_X;
static unsigned char Y = DEFAULT_Y;

static bool reversed = false;

extern void move_ships (void) {
    auto char symbol;
    generate_ship();

    while(true) {

        print_map();
        symbol = getch();

        switch (toupper(symbol)) {

            case 'A': move_left(); break;
            case 'D': move_right(); break;
            case 'W': move_top(); break;
            case 'S': move_bottom(); break;
            case 'R': reverse_ship(); break;

            case 13 : 
                if (set_ship()) {
                    exit_mode = false;
                    return;
                }
                else break;

            case 27: exit_mode = true; return;
            default: break;
        }
    }
}

static void generate_ship (void) {
    auto unsigned char counter;

    for (counter = 0; counter < size_ship; counter++) {
        if (map[Y+counter][X] == SET_BLOCK)
            map[Y+counter][X] = SET_BLOCK;
        else map[Y+counter][X] = BLOCK;
    }
}

static void reverse_ship (void) {
    auto unsigned char counter;

    if (reversed) {
        if (Y+size_ship <= 11) {
            for (counter = 1; counter < size_ship; counter++) {
                if (map[Y][X+counter*2] != SET_BLOCK)
                    map[Y][X+counter*2] = SPACE;

                if (map[Y+counter][X] == SET_BLOCK)
                    map[Y+counter][X] = SET_BLOCK;
                else map[Y+counter][X] = BLOCK;
            }
            reversed = false;
        }
    } else {
        if (X+(size_ship-1)*2 <= 21) {
            for (counter = 1; counter < size_ship; counter++) {
                if (map[Y+counter][X] != SET_BLOCK)
                    map[Y+counter][X] = SPACE;

                if (map[Y][X+counter*2] == SET_BLOCK)
                    map[Y][X+counter*2] = SET_BLOCK;
                else map[Y][X+counter*2] = BLOCK;
            }
            reversed = true;
        }
    }
}

static void move_left (void) {
    auto unsigned char counter;

    if (reversed) {
        if (X != 3) {
            X -= 2;

            if (map[Y][X+size_ship*2] != SET_BLOCK)
                map[Y][X+size_ship*2] = SPACE;

            if (map[Y][X] == SET_BLOCK)
                map[Y][X] = SET_BLOCK;
            else map[Y][X] = BLOCK;
        }
    } else {
        for (counter = 0; counter < size_ship; ++counter)
            if (map[Y+counter][X] != SET_BLOCK)
                map[Y+counter][X] = SPACE;

        if (X != 3) X -= 2;

        for (counter = 0; counter < size_ship; ++counter)
            if (map[Y+counter][X] == SET_BLOCK)
                map[Y+counter][X] = SET_BLOCK;
            else map[Y+counter][X] = BLOCK;
    }
}

static void move_right (void) {
    auto unsigned char counter;

    if (reversed) {
        if (X+(size_ship-1)*2 != 21) {
            if (map[Y][X] != SET_BLOCK)
                map[Y][X] = SPACE;

            if (map[Y][X+size_ship*2] == SET_BLOCK)
                map[Y][X+size_ship*2] = SET_BLOCK;
            else map[Y][X+size_ship*2] = BLOCK;

            X += 2;
        }
    } else {
        for (counter = 0; counter < size_ship; ++counter)
            if (map[Y+counter][X] != SET_BLOCK)
                map[Y+counter][X] = SPACE;

        if (X != 21) X += 2;

        for (counter = 0; counter < size_ship; ++counter)
            if (map[Y+counter][X] == SET_BLOCK)
                map[Y+counter][X] = SET_BLOCK;
            else map[Y+counter][X] = BLOCK;
    }
}

static void move_top (void) {
    auto unsigned char counter;

    if (reversed) {
        for (counter = 0; counter < size_ship; ++counter)
            if (map[Y][X+counter*2] != SET_BLOCK)
                map[Y][X+counter*2] = SPACE;

        if (Y != 1) --Y;

        for (counter = 0; counter < size_ship; ++counter)
            if (map[Y][X+counter*2] == SET_BLOCK)
                map[Y][X+counter*2] = SET_BLOCK;
            else map[Y][X+counter*2] = BLOCK;

    } else {
        if (Y != 1) {
            --Y;
            if (map[Y+size_ship][X] != SET_BLOCK)
                map[Y+size_ship][X] = SPACE;

            if (map[Y][X] == SET_BLOCK)
                map[Y][X] = SET_BLOCK;
            else map[Y][X] = BLOCK;
        }
    }
}

static void move_bottom (void) {
    auto unsigned char counter;

    if (reversed) {
        for (counter = 0; counter < size_ship; ++counter)
            if (map[Y][X+counter*2] != SET_BLOCK)
                map[Y][X+counter*2] = SPACE;

        if (Y != 10) ++Y;

        for (counter = 0; counter < size_ship; ++counter)
            if (map[Y][X+counter*2] == SET_BLOCK)
                map[Y][X+counter*2] = SET_BLOCK;
            else map[Y][X+counter*2] = BLOCK;

    } else {
        if (Y+size_ship != 11) {
            if (map[Y][X] != SET_BLOCK)
                map[Y][X] = SPACE;

            if (map[Y+size_ship][X] == SET_BLOCK)
                map[Y+size_ship][X] = SET_BLOCK;

            else map[Y+size_ship][X] = BLOCK;
            ++Y;
        }
    }
}

static bool set_ship (void) {
    auto unsigned char counter;

    if (reversed) {
        for (counter = 0; counter < size_ship; ++counter)
            if (map[Y][X+counter*2] == SET_BLOCK)
                return false;

        for (counter = 0; counter < size_ship; ++counter)
            map[Y][X+counter*2] = SET_BLOCK;

    } else {
        for (counter = 0; counter < size_ship; ++counter)
            if (map[Y+counter][X] == SET_BLOCK)
                return false;

        for (counter = 0; counter < size_ship; ++counter)
            map[Y+counter][X] = SET_BLOCK;
    }

    if (count_ship != 1) {
        --count_ship;
        switch(count_ship) {
            case 9: size_ship = 3; break;
            case 7: size_ship = 2; break;
            case 4: size_ship = 1; break;
            default: break;
        }

        X = DEFAULT_X;
        Y = DEFAULT_Y;

        reversed = false;
        generate_ship();

    } else return true;
    return false;
}

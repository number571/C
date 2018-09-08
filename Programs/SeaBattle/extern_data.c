#include "library.h"

#include <stdio.h>
#include <stdlib.h>

unsigned char size_ship = 4;
unsigned char count_ship = 10;

bool menu_mode = true;
bool exit_mode = false;
char win_mode  = -1;

char map[MAP_Y][MAP_X] = {
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

#if defined(_WIN32) || defined(_WIN64) 
    #define WINDOWS
    #include <conio.h>

#elif defined(unix)
    #define UNIX
    extern char getch (void);
#else
    #error "Platform is not supported"
#endif

#ifdef UNIX
    extern char getch (void) {
        char ch;
        system("stty raw");
        ch = getchar();
        system("stty cooked");
        return ch;
    }
#endif

extern void clear (void) {
    #ifdef WINDOWS
        printf("\033[2J");
        printf("\033[0;0f");
    #else
        system("clear");
    #endif
}

extern void print_map (void) {
    unsigned char x, y;
    clear();
    for (y = 0; y < MAP_Y; y++) {
        for (x = 0; x < MAP_X; x++)
            printf("%c", map[y][x]);
        printf("\n");
    }
}

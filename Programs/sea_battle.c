/**********************************************/
/*               LIBRARIES BLOCK              */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>

/**********************************************/
/*               PLATFORM BLOCK               */
/**********************************************/

#if defined(_WIN32) || defined(_WIN64) 
    #define WINDOWS
    #include <conio.h>

#elif defined(unix)
    #define UNIX
    char getch (void);
#else
    #error "Platform is not supported"
#endif

#ifdef UNIX
    char getch (void) {
        char ch;
        system("stty raw");
        ch = getchar();
        system("stty cooked");
        return ch;
    }
#endif

void clear (void) {
    #ifdef WINDOWS
        printf("\033[2J");
        printf("\033[0;0f");
    #else
        system("clear");
    #endif
}

/**********************************************/
/*                 MAPS BLOCK                 */
/**********************************************/

#define SET_BLOCK '#'
#define TARGET '+'
#define BLOCK '@'
#define SPACE ' '
#define HIT  'X' 
#define MISS '~'

#define MAP_X 52
#define MAP_Y 11

#define DEFAULT_X 3
#define DEFAULT_Y 1

#define DEFAULT_X_OPP 32
#define DEFAULT_Y_OPP 1

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

#define OPP_MAP_X 52
#define OPP_MAP_Y 11

char map_opp[OPP_MAP_Y][OPP_MAP_X] = {
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

#define MENU_X 16
#define MENU_Y 6

char menu[MENU_Y][MENU_X] = {
    " ============== ",
    " | START GAME | ",
    " ============== ",
    " -------------- ",
    " | STOP  GAME | ",
    " -------------- "
};

/**********************************************/
/*                 VARS BLOCK                 */
/**********************************************/

typedef enum {false, true} bool;

static bool menu_mode = true;
static bool reversed = false;
static bool exit_mode = false;

static char win;

static unsigned char X = DEFAULT_X;
static unsigned char Y = DEFAULT_Y;

static unsigned char X_opp = DEFAULT_X_OPP;
static unsigned char Y_opp = DEFAULT_Y_OPP;

static unsigned char opp_move_X = DEFAULT_X_OPP;
static unsigned char opp_move_Y = DEFAULT_Y_OPP;

static unsigned char size_ship = 4;
static unsigned char count_ship = 10;

static unsigned char count_damaged_ship = 0;
static unsigned char count_damaged_opp_ship = 0;

/**********************************************/
/*                 FUNC BLOCK                 */
/**********************************************/

/* MAIN BLOCK */
static void print_map (void);
static void print_result (const char* const s);

/* MENU BLOCK */
static void get_menu (void);
static void print_menu (void);
static void change_menu_mode (bool mode);

/* MOVE SHIPS BLOCK */
static void move_ships (void);
static void generate_ship (void);
static void reverse_ship (void);
static void move_left (void);
static void move_right (void);
static void move_top (void);
static void move_bottom (void);
static bool set_ship (void);

/* GAME BLOCK */
static void start_game (void);
static void generate_opp_map (void);
static bool boom_opp (void);
static char boom (void);
static void check_space (void);
static void check_target (void);
static void move_left_opp (void);
static void move_right_opp (void);
static void move_top_opp (void);
static void move_bottom_opp (void);

/**********************************************/
/*                 MAIN BLOCK                 */
/**********************************************/

int main (void) {
    srand(time(NULL));

    get_menu();
    if (menu_mode) {
        move_ships();

        if (!exit_mode) {
            start_game();

            if (win == 1) print_result("WIN ");
            else if (win == -1) print_result("EXIT");
            else print_result("LOSE");

        } else print_result("EXIT");
    } else clear();

    return 0;
}

static void print_result (const char* const s) {
    printf("\n\n\n /*************************/"\
           "\n /*      RESULT: %s     */"\
           "\n /*************************/\n\n\n", s);
}

static void print_map (void) {
    auto unsigned char x, y;
    clear();
    for (y = 0; y < MAP_Y; y++) {
        for (x = 0; x < MAP_X; x++)
            printf("%c", map[y][x]);
        printf("\n");
    }
}

/**********************************************/
/*                 MENU BLOCK                 */
/**********************************************/

static void get_menu (void) {
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

/**********************************************/
/*              MOVE SHIPS BLOCK              */
/**********************************************/

static void move_ships (void) {
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
                if (set_ship()) return;
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

/**********************************************/
/*                 GAME BLOCK                 */
/**********************************************/

static void start_game (void) {
    auto char symbol;
    auto char result;

    size_ship = 4;
    count_ship = 10;

    generate_opp_map();
    map[Y_opp][X_opp] = TARGET;

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

            case 27: { win = -1; return; }
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

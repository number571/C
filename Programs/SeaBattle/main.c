#include "library.h"
#include <stdio.h>


extern void clear (void);
extern void get_menu (void);
extern void move_ships (void);
extern void start_game (bool s);

extern bool menu_mode;
extern bool exit_mode;
extern char win_mode;


static void print_result (char s) {
    printf("\n\n"\
        "\n /*************************/"\
        "\n /*      RESULT: %s     */"\
        "\n /*************************/"\
    "\n\n\n", s ? "WIN " : "LOSE");
}

int main (void) {
    while (true) {
        get_menu();
        if (menu_mode) {
            move_ships();

            if (!exit_mode) {
                start_game(true);
                while (win_mode == -1) {
                    get_menu();
                    if (!menu_mode) 
                        goto end;
                    start_game(false);
                }
                print_result(win_mode);
                break;

            } else continue;
        } else break;
    }

end: 
    if (!menu_mode) clear();
    return 0;
}

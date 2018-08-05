#include <stdio.h>
#include <stdlib.h>

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

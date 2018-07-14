#!/usr/bin/tcc -run

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>

#define TRUE 1
#define N 20

struct {
    unsigned short int x;
    unsigned short int y;
} point, eat;

char map[N/2][N] = {
    "#...................",
    "....................",
    "....................",
    "....................",
    "....................",
    "....................",
    "....................",
    "....................",
    "....................",
    "...................."
};

void printMatrix (void);
void startGame (void);
void getMenu (void);
void dropEat (void);

int main (void) {

    getMenu();
    
    return 0;
}

void getMenu (void) {
    srand(time(NULL));

    char start_mode;
    start:

        system("clear");

        printf("\n");
        printf("\tPRESS '1' TO START\n");
        printf("\tPRESS '0' TO STOP\n");
        printf("\t> ");

        system("stty raw"); 
            start_mode = getchar();
        system("stty cooked"); 

    switch(toupper(start_mode)) {
        case '1': startGame(); break;
        case '0': printf("\n"); break;
        default: goto start;
    }

}

void dropEat (void) {
    unsigned short int x, y;
    again:
        x = eat.x;
        y = eat.y;
        eat.x = rand()%20;
        eat.y = rand()%10;
    if (eat.x == x && eat.y == y)
        goto again;
    map[eat.y][eat.x] = '*';
}

void startGame (void) {
    char command;
    point.x = point.y = 0;
    dropEat();
    while(TRUE) {
        system("clear");
        printMatrix();

        system("stty raw"); 
            command = getchar();
        system("stty cooked"); 

        switch(toupper(command)) {
            case '0': 
                return;
            case 'W': 
                if (point.y != 0) {
                    map[point.y][point.x] = '.';
                    --point.y;
                    map[point.y][point.x] = '#';
                }
                break;
            case 'S':
                if (point.y != 9) {
                    map[point.y][point.x] = '.';
                    ++point.y;
                    map[point.y][point.x] = '#';
                }
                break;
            case 'A':
                if (point.x != 0) {
                    map[point.y][point.x] = '.';
                    --point.x;
                    map[point.y][point.x] = '#';
                }
                break;
            case 'D':
                if (point.x != 19) {
                    map[point.y][point.x] = '.';
                    ++point.x;
                    map[point.y][point.x] = '#';
                }
                break;
        }

        if (point.y == eat.y && point.x == eat.x) {
            map[point.y][point.x] = '#';
            dropEat();
        }
    }
}

void printMatrix (void) {
    unsigned short x, y;
    for (y = 0; y < N/2; y++)  {
        for (x = 0; x < N; x++) 
            printf("%c", map[y][x]);
        printf("\n");
    }
}

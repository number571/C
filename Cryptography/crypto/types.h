#pragma once

typedef struct {
	char x; 
	char y;
} Point;

typedef enum {
	false, 
	true
} bool;

typedef union {
    unsigned char byte;
    struct {
        unsigned char _0: 1;
        unsigned char _1: 1;
        unsigned char _2: 1;
        unsigned char _3: 1;
        unsigned char _4: 1;
        unsigned char _5: 1;
        unsigned char _6: 1;
        unsigned char _7: 1;
    } bit;
} Byte;
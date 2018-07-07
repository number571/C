#include <stdio.h>

typedef unsigned char uint8_t;

union code {
    uint8_t number;
    struct {
        unsigned _0: 1;
        unsigned _1: 1;
        unsigned _2: 1;
        unsigned _3: 1;
        unsigned _4: 1;
        unsigned _5: 1;
        unsigned _6: 1;
        unsigned _7: 1;
    } bit;
};

int main(void) {
    union code check;
    check.number = 22;

    printf("%d %d %d %d %d %d %d %d\n", 
        check.bit._7, check.bit._6, check.bit._5, check.bit._4,
        check.bit._3, check.bit._2, check.bit._1, check.bit._0);

    return 0;
}

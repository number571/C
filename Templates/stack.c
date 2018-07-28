#include <stdio.h>
#define LIMIT 255

double buffer[LIMIT];
unsigned char count = 0;

void push (double num);
double pop (void);

int main (void) {
    push(3);
    push(7);
    push(9);

    printf("%lf\n", pop());
    printf("%lf\n", pop());
    printf("%lf\n", pop());
    return 0;
}

void push (double num) {
    if (count != LIMIT)
        buffer[count++] = num;
    else
        printf("Buffer is full\n");
}

double pop (void) {
    if (count != 0)
        return buffer[--count];
    else {
        printf("Buffer is void\n");
        return 0.0
    }
}

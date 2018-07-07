#include <stdio.h>
#define MAX 100

void push(int symbol);
int pop(void);

int buffer[MAX], stack = 0;

int main(void) {

    push(3); 
    push(5);
    push(7);

    printf("%d\n", pop());
    printf("%d\n", pop());
    printf("%d\n", pop());

    return 0;
}

void push(int symbol) {
    if (stack < MAX) 
        buffer[stack++] = symbol;
    else 
        printf("Error: stack overflow!\n");
}

int pop(void) {
    if (stack > 0) 
        return buffer[--stack]; 
    else 
        printf("Error: stack = 0!\n");
}

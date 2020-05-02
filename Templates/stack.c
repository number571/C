#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct Stack {
    uint32_t *buffer;
    size_t pointer;
    size_t size;
} Stack;

extern Stack *new_stack(size_t size);
extern void free_stack(Stack *stack);

extern void push_stack(Stack *stack, uint32_t value);
extern int32_t pop_stack(Stack *stack);

int main(void) {
    Stack *stack = new_stack(512);

    push_stack(stack, 5);
    push_stack(stack, 10);
    push_stack(stack, 20);

    printf("%d\n", pop_stack(stack));
    printf("%d\n", pop_stack(stack));
    printf("%d\n", pop_stack(stack));

    free_stack(stack);
    return 0;
}

extern Stack *new_stack(size_t size) {
    Stack *stack = (Stack*)malloc(sizeof(Stack));
    stack->buffer = (uint32_t*)malloc(size * sizeof(uint32_t));
    stack->pointer = 0;
    stack->size = size;
    return stack;
};

extern void push_stack(Stack *stack, uint32_t value) {
    if (stack->pointer == stack->size) {
        fprintf(stderr, "%s\n", "stack overflow");
        return;
    }
    stack->buffer[stack->pointer++] = value;
}

extern int32_t pop_stack(Stack *stack) {
    if (stack->pointer == 0) {
        fprintf(stderr, "%s\n", "stack overflow");
        return -1;
    }
    return stack->buffer[--stack->pointer];
}

extern void free_stack(Stack *stack) {
    free(stack->buffer);
    free(stack);
}

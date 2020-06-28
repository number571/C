#ifndef EXTCLIB_STACK_H_
#define EXTCLIB_STACK_H_

#include <stddef.h>
#include "type.h"

typedef struct Stack Stack;

extern Stack *new_stack(size_t size, vtype_t value);
extern void free_stack(Stack *stack);

extern size_t size_stack(Stack *stack);

extern void set_stack(Stack *stack, size_t index, void *value);
extern value_t get_stack(Stack *stack, size_t index);

extern void push_stack(Stack *stack, void *value);
extern value_t pop_stack(Stack *stack);

#endif /* EXTCLIB_STACK_H_ */

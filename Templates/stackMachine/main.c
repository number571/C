#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

#include "extclib/hashtab.h"
#include "extclib/stack.h"

#define OPERATION_NUM 18

typedef enum opcode_t {
    PUSH_CODE,
    POP_CODE,
    LABEL_CODE,
    JMP_CODE,
    JL_CODE,
    JLE_CODE,
    JG_CODE,
    JGE_CODE,
    JE_CODE,
    JNE_CODE,
    ADD_CODE,
    SUB_CODE,
    MUL_CODE,
    DIV_CODE,
    STORE_CODE,
    LOAD_CODE,
    STACK_CODE,
    PRINT_CODE,
    PASS_CODE, // code undefined
} opcode_t;

const char *codes[OPERATION_NUM] = {
    [PUSH_CODE]     = "push",
    [POP_CODE]      = "pop",
    [LABEL_CODE]    = "label",
    [JMP_CODE]      = "jump",
    [JL_CODE]       = "jl",
    [JLE_CODE]      = "jle",
    [JG_CODE]       = "jg",
    [JGE_CODE]      = "jge",
    [JE_CODE]       = "je",
    [JNE_CODE]      = "jne",
    [ADD_CODE]      = "add",
    [SUB_CODE]      = "sub",
    [MUL_CODE]      = "mul",
    [DIV_CODE]      = "div",
    [STORE_CODE]    = "store",
    [LOAD_CODE]     = "load",
    [STACK_CODE]    = "stack",
    [PRINT_CODE]    = "print",
};

extern int32_t open_sm(char *filename);
extern int32_t read_sm(FILE *file);

static char *_readcode(char *line, FILE *file, opcode_t *code);
static _Bool _isspace(char ch);

int main(void) {
    int res = open_sm("main.sm");
    printf("%d\n", res);
    return 0;
}

extern int32_t open_sm(char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "%s\n", "error: read file");
        return 0;
    }
    int32_t result = read_sm(file);
    fclose(file);
    return result;
}

extern int32_t read_sm(FILE *file) {
    HashTab *hashtab = new_hashtab(250, STRING_TYPE, DECIMAL_TYPE);
    char buffer[BUFSIZ] = {0};
    char *line;
    opcode_t code;

    // read labels
    while(fgets(buffer, BUFSIZ, file) != NULL) {
        line = _readcode(buffer, file, &code);
        switch(code) {
            case LABEL_CODE:
                set_hashtab(hashtab, string(line), decimal((int32_t)ftell(file)));
            break;
            default: ;
        }
    }

    Stack *stack = new_stack(10000, DECIMAL_TYPE);
    int32_t value = 0;
    fseek(file, 0, SEEK_SET);
    
    // read commands
    while(fgets(buffer, BUFSIZ, file) != NULL) {
        line = _readcode(buffer, file, &code);
        switch(code) {
            case STACK_CODE:
                push_stack(stack, decimal((int32_t)size_stack(stack)));
            break;
            case PRINT_CODE:
                value = pop_stack(stack).decimal;
                printf("%d\n", value);
                push_stack(stack, decimal(value));
            break;
            case STORE_CODE:
                if (line[0] == '$') {
                    char *arg = line + strlen(line) + 1;
                    while(isspace(*arg)) {
                        ++arg;
                    }
                    char *ptr = arg;
                    while(!isspace(*ptr)) {
                        ++ptr;
                    }
                    *ptr = '\0';
                    if (arg[0] == '$') {
                        int32_t value = (int32_t)atoi(arg+1);
                        set_stack(stack, atoi(line+1), decimal(get_stack(stack, value).decimal));
                        break;
                    }
                    int32_t value = (int32_t)atoi(arg);
                    set_stack(stack, atoi(line+1), decimal(value));
                }
            break;
            case LOAD_CODE:
                if (line[0] == '$') {
                    push_stack(stack, decimal(get_stack(stack, atoi(line+1)).decimal));
                }
            break;
            case PUSH_CODE:
                push_stack(stack, decimal(atoi(line)));
            break;
            case POP_CODE: 
                value = pop_stack(stack).decimal;
            break;
            case ADD_CODE: case SUB_CODE: case MUL_CODE: case DIV_CODE: {
                int32_t x = pop_stack(stack).decimal;
                int32_t y = pop_stack(stack).decimal;
                switch(code) {
                    case ADD_CODE:
                        x += y;
                    break;
                    case SUB_CODE:
                        x -= y;
                    break;
                    case MUL_CODE:
                        x *= y;
                    break;
                    case DIV_CODE:
                        x /= y;
                    break;
                    default: ;
                }
                push_stack(stack, decimal(x));
            }
            break;
            case JMP_CODE:
                value = get_hashtab(hashtab, line).decimal;
                fseek(file, value, SEEK_SET);
            break;
            case JL_CODE: case JLE_CODE: case JG_CODE: case JGE_CODE: case JE_CODE: case JNE_CODE: {
                value = get_hashtab(hashtab, line).decimal;
                int32_t x = pop_stack(stack).decimal;
                int32_t y = pop_stack(stack).decimal;
                switch(code) {
                    case JL_CODE:
                        if (x < y) {
                            fseek(file, value, SEEK_SET);
                        }
                    break;
                    case JLE_CODE:
                        if (x <= y) {
                            fseek(file, value, SEEK_SET);
                        }
                    break;
                    case JG_CODE:
                        if (x > y) {
                            fseek(file, value, SEEK_SET);
                        }
                    break;
                    case JGE_CODE:
                        if (x >= y) {
                            fseek(file, value, SEEK_SET);
                        }
                    break;
                    case JE_CODE:
                        if (x == y) {
                            fseek(file, value, SEEK_SET);
                        }
                    break;
                    case JNE_CODE:
                        if (x != y) {
                            fseek(file, value, SEEK_SET);
                        }
                    break;
                    default: ;
                }
            }
            break;
            default: ;
        }
    }
    free_stack(stack);
    free_hashtab(hashtab);
    return value;
}

static char *_readcode(char *line, FILE *file, opcode_t *code) {
    // pass spaces
    char *ptr = line;
    while(isspace(*ptr)) {
        ++ptr;
    }

    // read operator
    line = ptr;
    while(!_isspace(*ptr)) {
        ++ptr;
    }
    *ptr = '\0';

    // analyze operator
    *code = PASS_CODE;
    for (size_t i = 0; i < OPERATION_NUM; ++i) {
        if (strcmp(line, codes[i]) == 0) {
            *code = i;
            break;
        }
    }

    // pass spaces after operator
    ++ptr;
    while(isspace(*ptr)) {
        ++ptr;
    }

    // read argument
    line = ptr;
    while(!_isspace(*ptr)) {
        ++ptr;
    }
    *ptr = '\0';

    // return first argument
    return line;
}

static _Bool _isspace(char ch) {
    if (ch == '\0' || isspace(ch)) {
        return 1;
    }
    return 0;
}


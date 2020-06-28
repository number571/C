#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

#include "extclib/bigint.h"
#include "extclib/hashtab.h"
#include "extclib/stack.h"

#define OPERATION_NUM 19

/*
; Operators:
; 1. push, pop
; 2. add, sub, mul, div
; 3. jmp, je, jne, jl, jle, jg, jge
; 4. load, store
; 5. label, stack
*/

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
    COMMENT_CODE,
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
    [COMMENT_CODE]  = ";",
};

extern BigInt *open_sm(const char *filename);
extern BigInt *read_sm(FILE *file);

static char *_readcode(char *line, FILE *file, opcode_t *code);
static _Bool _strnull(char *str);
static _Bool _isspace(char ch);

int main(int argc, char const *argv[]) {
    if (argc < 2) {
        return 1;
    }
    BigInt *res = open_sm(argv[1]);
    if (res == NULL) {
        return 2;
    }
    println_bigint(res);
    free_bigint(res);
    return 0;
}

extern BigInt *open_sm(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "%s\n", "error: read file");
        return NULL;
    }
    BigInt *result = read_sm(file);
    fclose(file);
    return result;
}

extern BigInt *read_sm(FILE *file) {
    HashTab *hashtab = new_hashtab(250, STRING_TYPE, DECIMAL_TYPE);
    char buffer[BUFSIZ] = {0};
    size_t line_index = 0;
    _Bool err_exist = 0;
    char *line;
    opcode_t code;

    // read labels, check syntax
    while(fgets(buffer, BUFSIZ, file) != NULL) {
        ++line_index;
        line = _readcode(buffer, file, &code);
        if((code == PASS_CODE && _strnull(line)) || code == COMMENT_CODE) {
            continue;
        }
        if (code == PASS_CODE) {
            err_exist = 1;
            fprintf(stderr, "error: line %ld\n", line_index);
        }
        switch(code) {
            case LABEL_CODE:
                set_hashtab(hashtab, string(line), decimal((int32_t)ftell(file)));
            break;
            default: ;
        }
    }
    if (err_exist) {
        free_hashtab(hashtab);
        return NULL;
    }

    Stack *stack = new_stack(10000, BIGINT_TYPE);
    BigInt *value = new_bigint("0");
    fseek(file, 0, SEEK_SET);

    // read commands
    while(fgets(buffer, BUFSIZ, file) != NULL) {
        line = _readcode(buffer, file, &code);
        switch(code) {
            case STACK_CODE: {
                BigInt *num = new_bigint("0");
                cpynum_bigint(num, (uint32_t)size_stack(stack));
                push_stack(stack, num);
            }
            break;
            case PRINT_CODE: {
                BigInt *num = pop_stack(stack).bigint;
                println_bigint(num);
                push_stack(stack, num);
            }
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
                        int32_t index = (int32_t)atoi(arg+1);
                        BigInt *num = new_bigint("0");
                        cpy_bigint(num, get_stack(stack, index).bigint);
                        index = atoi(line+1);
                        BigInt *x = get_stack(stack, index).bigint;
                        free_bigint(x);
                        set_stack(stack, index, num);
                        break;
                    }
                    BigInt *num = new_bigint("0");
                    cpynum_bigint(num, (int32_t)atoi(arg));
                    int32_t index = atoi(line+1);
                    BigInt *x = get_stack(stack, index).bigint;
                    free_bigint(x);
                    set_stack(stack, index, num);
                }
            break;
            case LOAD_CODE:
                if (line[0] == '$') {
                    BigInt *num = dup_bigint(get_stack(stack, atoi(line+1)).bigint);
                    push_stack(stack, num);
                }
            break;
            case PUSH_CODE: {
                BigInt *num = new_bigint("0");
                cpynum_bigint(num, (uint32_t)atoi(line));
                push_stack(stack, num);
            }
            break;
            case POP_CODE: {
                BigInt *num = pop_stack(stack).bigint;
                cpy_bigint(value, num);
                free_bigint(num);
            }
            break;
            case ADD_CODE: case SUB_CODE: case MUL_CODE: case DIV_CODE: {
                BigInt *x = pop_stack(stack).bigint;
                BigInt *y = pop_stack(stack).bigint;
                switch(code) {
                    case ADD_CODE:
                        add_bigint(x, x, y);
                    break;
                    case SUB_CODE:
                        sub_bigint(x, x, y);
                    break;
                    case MUL_CODE:
                        mul_bigint(x, x, y);
                    break;
                    case DIV_CODE:
                        div_bigint(x, x, y);
                    break;
                    default: ;
                }
                push_stack(stack, x);
                free_bigint(y);
            }
            break;
            case JMP_CODE: {
                int32_t index = get_hashtab(hashtab, line).decimal;
                fseek(file, index, SEEK_SET);
            }
            break;
            case JL_CODE: case JLE_CODE: case JG_CODE: case JGE_CODE: case JE_CODE: case JNE_CODE: {
                int32_t index = get_hashtab(hashtab, line).decimal;
                BigInt *x = pop_stack(stack).bigint;
                BigInt *y = pop_stack(stack).bigint;
                switch(code) {
                    case JL_CODE:
                        if (cmp_bigint(x, y) < 0) {
                            fseek(file, index, SEEK_SET);
                        }
                    break;
                    case JLE_CODE:
                        if (cmp_bigint(x, y) <= 0) {
                            fseek(file, index, SEEK_SET);
                        }
                    break;
                    case JG_CODE:
                        if (cmp_bigint(x, y) > 0) {
                            fseek(file, index, SEEK_SET);
                        }
                    break;
                    case JGE_CODE:
                        if (cmp_bigint(x, y) >= 0) {
                            fseek(file, index, SEEK_SET);
                        }
                    break;
                    case JE_CODE:
                        if (cmp_bigint(x, y) == 0) {
                            fseek(file, index, SEEK_SET);
                        }
                    break;
                    case JNE_CODE:
                        if (cmp_bigint(x, y) != 0) {
                            fseek(file, index, SEEK_SET);
                        }
                    break;
                    default: ;
                }
                free_bigint(y);
                free_bigint(x);
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

    // operators without args
    switch(*code) {
        case PASS_CODE:
        case POP_CODE:
        case STACK_CODE:
        case PRINT_CODE:
        case COMMENT_CODE:
        return line;
        default: ;
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

static _Bool _strnull(char *str) {
    while(isspace(*str)) {
        ++str;
    }
    if(*str == '\0') {
        return 1;
    }
    return 0;
}

static _Bool _isspace(char ch) {
    if (ch == '\0' || isspace(ch)) {
        return 1;
    }
    return 0;
}


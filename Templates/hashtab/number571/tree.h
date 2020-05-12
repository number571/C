#pragma once

#include <stdint.h>

typedef enum {
    DECIMAL_ELEM,
    REAL_ELEM,
    STRING_ELEM,
} vtype_tree_t;

typedef union {
    int64_t decimal;
    double real;
    uint8_t *string;
} value_tree_t;

typedef struct tree_node {
    struct {
        value_tree_t key;
        value_tree_t value;
    } data;
    struct tree_node *left;
    struct tree_node *right;
    struct tree_node *parent;
} tree_node;

typedef struct Tree {
    struct {
        vtype_tree_t key;
        vtype_tree_t value;
    } type;
    struct tree_node *node;
} Tree;

extern Tree *new_tree(vtype_tree_t key, vtype_tree_t value);
extern void free_tree(Tree *tree);

extern value_tree_t get_tree(Tree *tree, void *key);
extern void set_tree(Tree *tree, void *key, void *value);
extern void del_tree(Tree *tree, void *key);
extern _Bool in_tree(Tree *tree, void *key);

extern void *decimal(int64_t x);
extern void *string(uint8_t *x);
extern void *real(double x);

extern void print_tree(Tree *tree);
extern void print_tree_as_list(Tree *tree);

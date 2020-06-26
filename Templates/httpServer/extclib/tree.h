#ifndef EXTCLIB_TREE_H_
#define EXTCLIB_TREE_H_

#include <stddef.h>
#include <stdint.h>

#include "type.h"

typedef struct Tree Tree;

extern Tree *new_tree(vtype_t key, vtype_t value);
extern void free_tree(Tree *tree);

extern value_t get_tree(Tree *tree, void *key);
extern int8_t set_tree(Tree *tree, void *key, void *value);
extern void del_tree(Tree *tree, void *key);
extern _Bool in_tree(Tree *tree, void *key);

extern _Bool eq_tree(Tree *x, Tree *y);
extern size_t size_tree(Tree *tree);
extern size_t sizeof_tree(void);

extern void print_tree(Tree *tree);
extern void println_tree(Tree *tree);

extern void print_tree_branches(Tree *tree);
extern void println_tree_branches(Tree *tree);

#endif /* EXTCLIB_TREE_H_ */

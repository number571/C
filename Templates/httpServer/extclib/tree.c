#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "tree.h"
#include "type.h"

typedef struct tree_node {
    struct {
        value_t key;
        value_t value;
    } data;
    _Bool exist;
    struct tree_node *left;
    struct tree_node *right;
    struct tree_node *parent;
} tree_node;

typedef struct Tree {
    struct {
        vtype_t key;
        vtype_t value;
    } type;
    size_t size;
    struct tree_node *node;
} Tree;

static tree_node *_new_node(vtype_t tkey, vtype_t tvalue, void *key, void *value);
static void _set_tree(Tree *tree, tree_node *node, vtype_t tkey, vtype_t tvalue, void *key, void *value);
static void _set_key(tree_node *node, vtype_t tkey, void *key);
static void _set_value(tree_node *node, vtype_t tvalue, void *value);
static void _free_tree(Tree *tree, tree_node *node);
static void _free_key_tree(vtype_t type, tree_node *node);
static void _free_value_tree(vtype_t type, tree_node *node);
static void _print_tree(Tree *tree, tree_node *node, vtype_t tkey, vtype_t tvalue);
static void _print_branches_tree(Tree *tree, tree_node *node, vtype_t tkey, vtype_t tvalue);
static void _print_node_tree(Tree *tree, tree_node *node, vtype_t tkey, vtype_t tvalue);
static tree_node *_get_tree(tree_node *node, vtype_t tkey, void *key);
static int8_t _cmp_tkey_tree(tree_node *node, vtype_t tkey, void *key);
static int8_t _cmp_int32(int32_t x, int32_t y);
static tree_node *_del1_tree(Tree *tree, vtype_t tkey, void *key);
static void _del2_tree(Tree *tree, tree_node *node);
static void _del3_tree(Tree *tree, tree_node *node);
static _Bool _eq_tree(vtype_t tkey, vtype_t tvalue, tree_node *x, tree_node *y);

extern Tree *new_tree(vtype_t key, vtype_t value) {
    switch(key){
        case DECIMAL_TYPE: 
        case STRING_TYPE:
            break;
        default:
            fprintf(stderr, "%s\n", "key type not supported");
            return NULL;
    }
    switch(value) {
        case DECIMAL_TYPE: 
        case REAL_TYPE: 
        case STRING_TYPE: 
            break;
        default:
            fprintf(stderr, "%s\n", "value type not supported");
            return NULL;
    }
    Tree *tree = (Tree*)malloc(sizeof(Tree));
    tree->type.key = key;
    tree->type.value = value;
    tree->node = NULL;
    tree->size = 0;
    return tree;
}

extern void free_tree(Tree *tree) {
    _free_tree(tree, tree->node);
    free(tree);
}

extern _Bool in_tree(Tree *tree, void *key) {
    return _get_tree(tree->node, tree->type.key, key) != NULL;
}

extern value_t get_tree(Tree *tree, void *key) {
    tree_node *node = _get_tree(tree->node, tree->type.key, key);
    if (node == NULL) {
        fprintf(stderr, "%s\n", "value undefined");
        value_t none = {
            .decimal = 0,
        };
        return none;
    }
    return node->data.value;
}

extern int8_t set_tree(Tree *tree, void *key, void *value) {
    if (tree->node == NULL) {
        tree->node = _new_node(tree->type.key, tree->type.value, key, value);
        tree->size += 1;
        return 0;
    }
    _set_tree(tree, tree->node, tree->type.key, tree->type.value, key, value);
    return 0;
}

extern void del_tree(Tree *tree, void *key) {
    tree_node *node = _del1_tree(tree, tree->type.key, key);
    if (node == NULL) {
        return;
    }
    if (node->left != NULL && node->right != NULL) {
        _del3_tree(tree, node);
        return;
    }
    _del2_tree(tree, node);
    return;
}

extern _Bool eq_tree(Tree *x, Tree *y) {
    if (x->type.key != y->type.key) {
        return 0;
    }
    if (x->type.value != y->type.value) {
        return 0;
    }
    if (x->size != y->size) {
        return 0;
    }
    return _eq_tree(x->type.key, x->type.value, x->node, y->node);
}

static _Bool _eq_tree(vtype_t tkey, vtype_t tvalue, tree_node *x, tree_node *y) {
    if (x == NULL && y == NULL) {
        return 1;
    }
    if (x != NULL && y != NULL) {
        _Bool fkey = 0;
        _Bool fval = 0;
        switch(tkey) {
            case DECIMAL_TYPE:
                fkey = x->data.key.decimal == y->data.key.decimal;
            break;
            case STRING_TYPE:
                fkey = strcmp((char*)x->data.key.string, (char*)y->data.key.string) == 0;
            break;
            default: ;
        }
        switch(tvalue) {
            case DECIMAL_TYPE:
                fval = x->data.value.decimal == y->data.value.decimal;
            break;
            case REAL_TYPE:
                fval = x->data.value.real == y->data.value.real;
            break;
            case STRING_TYPE:
                fval = strcmp((char*)x->data.value.string, (char*)y->data.value.string) == 0;
            break;
            default: ;
        }
        return fkey && fval && _eq_tree(tkey, tvalue, x->left, y->left) && _eq_tree(tkey, tvalue, x->right, y->right);
    }
    return 0;
}

extern size_t size_tree(Tree *tree) {
    return tree->size;
}

extern size_t sizeof_tree(void) {
    return sizeof(Tree);
}

extern void print_tree(Tree *tree) {
    printf("#T[ ");
    _print_tree(tree, tree->node, tree->type.key, tree->type.value);
    putchar(']');
}

extern void println_tree(Tree *tree) {
    print_tree(tree);
    putchar('\n');
}

extern void print_tree_branches(Tree *tree) {
    _print_branches_tree(tree, tree->node, tree->type.key, tree->type.value);
}

extern void println_tree_branches(Tree *tree) {
    _print_branches_tree(tree, tree->node, tree->type.key, tree->type.value);
    putchar('\n');
}

static tree_node *_new_node(vtype_t tkey, vtype_t tvalue, void *key, void *value) {
    tree_node *node = (tree_node*)malloc(sizeof(tree_node));
    node->exist = 0;
    _set_key(node, tkey, key);
    _set_value(node, tvalue, value);
    node->left = NULL;
    node->right = NULL;
    node->parent = NULL;
    return node;
}

static void _set_key(tree_node *node, vtype_t tkey, void *key) {
    if (node->exist) {
        _free_key_tree(tkey, node);
    }
    switch(tkey) {
        case DECIMAL_TYPE:
            node->data.key.decimal = (int32_t)(intptr_t)key;
        break;
        case STRING_TYPE: {
            size_t size = strlen((char*)key);
            node->data.key.string = (char*)malloc(sizeof(char)*size+1);
            strcpy(node->data.key.string, (char*)key);
        }
        break;
        default: ;
    }
}

static void _set_value(tree_node *node, vtype_t tvalue, void *value) {
    if (node->exist) {
        _free_value_tree(tvalue, node);
    }
    switch(tvalue) {
        case DECIMAL_TYPE:
            node->data.value.decimal = (int32_t)(intptr_t)value;
        break;
        case REAL_TYPE:
            node->data.value.real = *(double*)value;
            free((double*)value);
        break;
        case STRING_TYPE: {
            size_t size = strlen((char*)value);
            node->data.value.string = (char*)malloc(sizeof(char)*size+1);
            strcpy(node->data.value.string, (char*)value);
        }
        break;
        default: ;
    }
    node->exist = 1;
}

static void _set_tree(Tree *tree, tree_node *node, vtype_t tkey, vtype_t tvalue, void *key, void *value) {
    int8_t cond = _cmp_tkey_tree(node, tkey, key);
    if (cond > 0) {
        if (node->right == NULL) {
            node->right = _new_node(tkey, tvalue, key, value);
            node->right->parent = node;
            tree->size += 1;
        } else {
            _set_tree(tree, node->right, tkey, tvalue, key, value);
        }
    } else if (cond < 0) {
        if (node->left == NULL) {
            node->left = _new_node(tkey, tvalue, key, value);
            node->left->parent = node;
            tree->size += 1;
        } else {
            _set_tree(tree, node->left, tkey, tvalue, key, value);
        }
    } else {
        _set_key(node, tkey, key);
        _set_value(node, tvalue, value);
    }
}

static tree_node *_get_tree(tree_node *node, vtype_t tkey, void *key) {
    if (node == NULL) {
        return NULL;
    }
    int8_t cond = _cmp_tkey_tree(node, tkey, key);
    if (cond > 0) {
        return _get_tree(node->right, tkey, key);
    } else if (cond < 0) {
        return _get_tree(node->left, tkey, key);
    }
    return node;
}

static int8_t _cmp_tkey_tree(tree_node *node, vtype_t tkey, void *key) {
    int8_t cond = 0;
    switch(tkey) {
        case DECIMAL_TYPE:
            cond = _cmp_int32((int32_t)(intptr_t)key, node->data.key.decimal);
        break;
        case STRING_TYPE:
            cond = strcmp((char*)key, node->data.key.string);
        break;
        default: ;
    }
    return cond;
}

static int8_t _cmp_int32(int32_t x, int32_t y) {
    if (x > y) {
        return 1;
    } else if (x < y) {
        return -1;
    }
    return 0;
}

static tree_node *_del1_tree(Tree *tree, vtype_t tkey, void *key) {
    tree_node *node = tree->node;
    node =  _get_tree(node, tkey, key);
    if (node == NULL) {
        return NULL;
    }
    if (node->left != NULL || node->right != NULL) {
        return node;
    }
    tree_node *parent = node->parent;
    if (parent == NULL) {
        tree->node = NULL;
    } else if (parent->left == node) {
        parent->left = NULL;
    } else {
        parent->right = NULL;
    }
    tree->size -= 1;
    _free_key_tree(tree->type.key, node);
    _free_value_tree(tree->type.value, node);
    free(node);
    return NULL;
}

static void _del2_tree(Tree *tree, tree_node *node) {
    tree_node *parent = node->parent;
    tree_node *temp;
    if (node->right != NULL) {
        temp = node->right;
    } else {
        temp = node->left;
    }
    if (parent == NULL) {
        tree->node = temp;
    } else if (parent->left == node) {
        parent->left = temp;
    } else {
        parent->right = temp;
    }
    tree->size -= 1;
    temp->parent = parent;
    _free_key_tree(tree->type.key, node);
    _free_value_tree(tree->type.value, node);
    free(node);
}

static void _del3_tree(Tree *tree, tree_node *node) {
    tree_node *ptr = node->right;
    while (ptr->left != NULL) {
        ptr = ptr->left;
    }
    node->data.key = ptr->data.key;
    node->data.value = ptr->data.value;
    tree_node *parent = ptr->parent;
    if (parent->left == ptr) {
        parent->left = NULL;
    } else {
        parent->right = NULL;
    }
    tree->size -= 1;
    // _free_key_tree(tree->type.key, node);
    // _free_value_tree(tree->type.value, ptr);
    free(ptr);
}

static void _print_node_tree(Tree *tree, tree_node *node, vtype_t tkey, vtype_t tvalue) {
    putchar('{');
    switch(tkey) {
        case DECIMAL_TYPE:
            printf("%d", node->data.key.decimal);
        break;
        case STRING_TYPE:
            printf("'%s'", node->data.key.string);
        break;
        default: ;
    }
    printf(" => ");
    switch(tvalue) {
        case DECIMAL_TYPE:
            printf("%d", node->data.value.decimal);
        break;
        case REAL_TYPE:
            printf("%lf", node->data.value.real);
        break;
        case STRING_TYPE:
            printf("'%s'", node->data.value.string);
        break;
        default: ;
    }
    printf("} ");
}

static void _print_branches_tree(Tree *tree, tree_node *node, vtype_t tkey, vtype_t tvalue) {
    if (node == NULL) {
        printf("null");
        return;
    }
    putchar('(');
    _print_branches_tree(tree, node->left, tkey, tvalue);
    putchar(' ');
    _print_node_tree(tree, node, tkey, tvalue);
    _print_branches_tree(tree, node->right, tkey, tvalue);
    putchar(')');
}

static void _print_tree(Tree *tree, tree_node *node, vtype_t tkey, vtype_t tvalue) {
    if (node == NULL) {
        return;
    }
    _print_tree(tree, node->left, tkey, tvalue);
    _print_node_tree(tree, node, tkey, tvalue);
    _print_tree(tree, node->right, tkey, tvalue);
}

static void _free_tree(Tree *tree, tree_node *node) {
    if (node == NULL) {
        return;
    }
    _free_tree(tree, node->left);
    _free_tree(tree, node->right);
    _free_key_tree(tree->type.key, node);
    _free_value_tree(tree->type.value, node);
    free(node);
}

static void _free_key_tree(vtype_t type, tree_node *node) {
    switch(type) {
        case STRING_TYPE:
            free(node->data.key.string);
        break;
        default: ;
    }
}

static void _free_value_tree(vtype_t type, tree_node *node) {
    switch(type) {
        case STRING_TYPE:
            free(node->data.value.string);
        break;
        default: ;
    }
}

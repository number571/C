#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "hashtab.h"
#include "tree.h"
#include "type.h"

typedef struct HashTab {
    struct {
        vtype_t key;
        vtype_t value;
    } type;
    size_t size;
    Tree **table;
} HashTab;

static uint32_t _get_hash(HashTab *hashtab, void *key);
static uint32_t _strhash(char *s, size_t size);

extern HashTab *new_hashtab(size_t size, vtype_t key, vtype_t value) {
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
    HashTab *hashtab = (HashTab*)malloc(sizeof(HashTab));
    hashtab->table = (Tree**)malloc(size * sizeof(Tree*));
    for (size_t i = 0; i < size; ++i) {
        hashtab->table[i] = new_tree(key, value);
    }
    hashtab->size = size;
    hashtab->type.key = key;
    hashtab->type.value = value;
    return hashtab;
}

extern void del_hashtab(HashTab *hashtab, void *key) {
    uint32_t hash = _get_hash(hashtab, key);
    del_tree(hashtab->table[hash], key);
}

extern _Bool in_hashtab(HashTab *hashtab, void *key) {
    uint32_t hash = _get_hash(hashtab, key);
    return in_tree(hashtab->table[hash], key);
}

extern value_t get_hashtab(HashTab *hashtab, void *key) {
    uint32_t hash = _get_hash(hashtab, key);
    return get_tree(hashtab->table[hash], key);
}

extern int8_t set_hashtab(HashTab *hashtab, void *key, void *value) {
    uint32_t hash = _get_hash(hashtab, key);
    set_tree(hashtab->table[hash], key, value);
    return 0;
}

extern _Bool eq_hashtab(HashTab *x, HashTab *y) {
    if (x->type.key != y->type.key) {
        return 0;
    }
    if (x->type.value != y->type.value) {
        return 0;
    }
    if (x->size != y->size) {
        return 0;
    }
    for (size_t i = 0; i < x->size; ++i) {
        if (size_tree(x->table[i]) != size_tree(y->table[i])) {
            return 0;
        }
        if (eq_tree(x->table[i], y->table[i]) != 0) {
            return 0;
        }
    }
    return 1;
}

extern size_t size_hashtab(HashTab *hashtab) {
    return hashtab->size;
}

extern size_t sizeof_hashtab(void) {
    return sizeof(HashTab);
}

extern void free_hashtab(HashTab *hashtab) {
    for (size_t i = 0; i < hashtab->size; ++i) {
        free_tree(hashtab->table[i]);
    }
    free(hashtab->table);
    free(hashtab);
}

extern void print_hashtab(HashTab *hashtab) {
    printf("#H[ ");
    for (size_t i = 0; i < hashtab->size; ++i) {
        if (size_tree(hashtab->table[i]) == 0) {
            continue;
        }
        printf("(%u :: ", (uint32_t)i);
        print_tree(hashtab->table[i]);
        printf(") ");
    }
    putchar(']');
}

extern void println_hashtab(HashTab *hashtab) {
    print_hashtab(hashtab);
    putchar('\n');
}

extern void print_hashtab_format(HashTab *hashtab) {
    printf("#H[\n");
    for (size_t i = 0; i < hashtab->size; ++i) {
        if (size_tree(hashtab->table[i]) == 0) {
            continue;
        }
        printf("\t(%u :: ", (uint32_t)i);
        print_tree(hashtab->table[i]);
        printf(")\n");
    }
    putchar(']');
}

extern void println_hashtab_format(HashTab *hashtab) {
    print_hashtab_format(hashtab);
    putchar('\n');
}

static uint32_t _get_hash(HashTab *hashtab, void *key) {
    uint32_t hash = 0;
    switch(hashtab->type.key) {
        case DECIMAL_TYPE:
            hash = (uint32_t)(intptr_t)key % hashtab->size;
        break;
        case STRING_TYPE:
            hash = _strhash((char*)key, hashtab->size);
        break;
        default: ;
    }
    return hash;
}

static uint32_t _strhash(char *s, size_t size) {
    uint32_t hashval;
    for (hashval = 0; *s != '\0'; ++s) {
        hashval = *s + 31 * hashval;
    }
    return hashval % size;
}

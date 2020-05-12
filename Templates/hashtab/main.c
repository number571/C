#include <stdio.h>
#include <stdlib.h>

#include "number571/tree.h"

typedef struct HashTab {
	struct {
        vtype_tree_t key;
        vtype_tree_t value;
    } type;
    size_t size;
    Tree **table;
} HashTab;

extern HashTab *new_hashtab(size_t size, vtype_tree_t key, vtype_tree_t value);
extern void free_hashtab(HashTab *hashtab);

extern value_tree_t get_hashtab(HashTab *hashtab, void *key);
extern void set_hashtab(HashTab *hashtab, void *key, void *value);
extern void del_hashtab(HashTab *hashtab, void *key);
extern _Bool in_hashtab(HashTab *hashtab, void *key);

extern void print_hashtab(HashTab *hashtab);

static uint32_t _strhash(uint8_t *s, size_t size);

int main(void) {
	HashTab *hashtab = new_hashtab(2, STRING_ELEM, DECIMAL_ELEM);

	set_hashtab(hashtab, string("A"), decimal(555));
	set_hashtab(hashtab, string("B"), decimal(333));
	set_hashtab(hashtab, string("C"), decimal(111));
	set_hashtab(hashtab, string("D"), decimal(777));

	del_hashtab(hashtab, string("A"));
	del_hashtab(hashtab, string("C"));

	print_hashtab(hashtab);
	free_hashtab(hashtab);
    return 0;
}

extern HashTab *new_hashtab(size_t size, vtype_tree_t key, vtype_tree_t value) {
	switch(key){
        case DECIMAL_ELEM: case STRING_ELEM:
            break;
        default:
            fprintf(stderr, "%s\n", "key type not supported");
            return NULL;
    }
    switch(value) {
        case DECIMAL_ELEM: case REAL_ELEM: case STRING_ELEM:
            break;
        default:
            fprintf(stderr, "%s\n", "value type not supported");
            return NULL;
    }
    HashTab *hashtab = (HashTab*)malloc(sizeof(HashTab));
    hashtab->table = (Tree**)malloc(size * sizeof(Tree));
    for (size_t i = 0; i < size; ++i) {
    	hashtab->table[i] = new_tree(key, value);
    }
    hashtab->size = size;
    hashtab->type.key = key;
    hashtab->type.value = value;
    return hashtab;
}

extern void del_hashtab(HashTab *hashtab, void *key) {
	uint32_t hash;
	switch(hashtab->type.key) {
		case DECIMAL_ELEM:
			hash = (uint64_t)key % hashtab->size;
		break;
		case STRING_ELEM:
			hash = _strhash((uint8_t*)key, hashtab->size);
		break;
	}
	del_tree(hashtab->table[hash], key);
}

extern _Bool in_hashtab(HashTab *hashtab, void *key) {
	uint32_t hash;
	_Bool result;
	switch(hashtab->type.key) {
		case DECIMAL_ELEM:
			hash = (uint64_t)key % hashtab->size;
		break;
		case STRING_ELEM:
			hash = _strhash((uint8_t*)key, hashtab->size);
		break;
	}
	result = in_tree(hashtab->table[hash], key);
	return result;
}

extern value_tree_t get_hashtab(HashTab *hashtab, void *key) {
	uint32_t hash;
	value_tree_t result;
	switch(hashtab->type.key) {
		case DECIMAL_ELEM:
			hash = (uint64_t)key % hashtab->size;
		break;
		case STRING_ELEM:
			hash = _strhash((uint8_t*)key, hashtab->size);
		break;
	}
	result = get_tree(hashtab->table[hash], key);
	return result;
}

extern void set_hashtab(HashTab *hashtab, void *key, void *value) {
	uint32_t hash;
	switch(hashtab->type.key) {
		case DECIMAL_ELEM:
			hash = (uint64_t)key % hashtab->size;
		break;
		case STRING_ELEM:
			hash = _strhash((uint8_t*)key, hashtab->size);
		break;
	}
	set_tree(hashtab->table[hash], key, value);
}

extern void free_hashtab(HashTab *hashtab) {
	for (size_t i = 0; i < hashtab->size; ++i) {
    	free_tree(hashtab->table[i]);
    }
    free(hashtab->table);
    free(hashtab);
}

extern void print_hashtab(HashTab *hashtab) {
	uint32_t hash;
	printf("{\n");
	for (size_t i = 0; i < hashtab->size; ++i) {
    	if (hashtab->table[i]->node == NULL) {
    		continue;
    	}
    	switch(hashtab->type.key) {
    		case DECIMAL_ELEM:
    			hash = hashtab->table[i]->node->data.key.decimal % hashtab->size;
    		break;
    		case STRING_ELEM:
    			hash = _strhash(hashtab->table[i]->node->data.key.string, hashtab->size);
    		break;
    	}
    	printf("\t%d => ", hash);
    	print_tree_as_list(hashtab->table[i]);
    }
    printf("}\n");
}

static uint32_t _strhash(uint8_t *s, size_t size) {
	uint32_t hashval;
	for (hashval = 0; *s != '\0'; ++s) {
		hashval = *s + 31 * hashval;
	}
	return hashval % size;
}

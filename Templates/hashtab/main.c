#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

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

extern _Bool in_hashtab(HashTab *hashtab, void *key);
extern void set_hashtab(HashTab *hashtab, void *key, void *value);
extern value_tree_t get_hashtab(HashTab *hashtab, void *key);
extern void del_hashtab(HashTab *hashtab, void *key);

extern void print_hashtab(HashTab *hashtab);

static uint32_t _strhash(uint8_t *s, size_t hashsize);

int main(void) {
	HashTab *hashtab = new_hashtab(10000, STRING_ELEM, DECIMAL_ELEM);

	set_hashtab(hashtab, string("hello"), decimal(11));
	set_hashtab(hashtab, string("world"), decimal(22));
	set_hashtab(hashtab, string("qwerty"), decimal(33));
	set_hashtab(hashtab, string("abcdef"), decimal(44));

	del_hashtab(hashtab, string("qwerty"));
	printf("%d\n", get_hashtab(hashtab, string("world")).decimal);

	print_hashtab(hashtab);
	free_hashtab(hashtab);
    return 0;
}

extern void del_hashtab(HashTab *hashtab, void *key) {
	uint32_t hash;
	switch(hashtab->type.key) {
		case DECIMAL_ELEM:
			hash = (uint64_t)key % hashtab->size;
			del_tree(hashtab->table[hash], key);
		break;
		case STRING_ELEM:
			hash = _strhash((uint8_t*)key, hashtab->size);
			del_tree(hashtab->table[hash], key);
		break;
	}
}

extern value_tree_t get_hashtab(HashTab *hashtab, void *key) {
	uint32_t hash;
	value_tree_t result;
	switch(hashtab->type.key) {
		case DECIMAL_ELEM:
			hash = (uint64_t)key % hashtab->size;
			result = get_tree(hashtab->table[hash], key);
		break;
		case STRING_ELEM:
			hash = _strhash((uint8_t*)key, hashtab->size);
			result = get_tree(hashtab->table[hash], key);
		break;
	}
	return result;
}

extern _Bool in_hashtab(HashTab *hashtab, void *key) {
	uint32_t hash;
	_Bool result;
	switch(hashtab->type.key) {
		case DECIMAL_ELEM:
			hash = (uint64_t)key % hashtab->size;
			result = in_tree(hashtab->table[hash], key);
		break;
		case STRING_ELEM:
			hash = _strhash((uint8_t*)key, hashtab->size);
			result = in_tree(hashtab->table[hash], key);
		break;
	}
	return result;
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

extern void set_hashtab(HashTab *hashtab, void *key, void *value) {
	uint32_t hash;
	switch(hashtab->type.key) {
		case DECIMAL_ELEM:
			hash = (uint64_t)key % hashtab->size;
			set_tree(hashtab->table[hash], key, value);
		break;
		case STRING_ELEM:
			hash = _strhash((uint8_t*)key, hashtab->size);
			set_tree(hashtab->table[hash], key, value);
		break;
	}
}

extern HashTab *new_hashtab(size_t size, vtype_tree_t key, vtype_tree_t value) {
	switch(key) {
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

extern void free_hashtab(HashTab *hashtab) {
	for (size_t i = 0; i < hashtab->size; ++i) {
		free_tree(hashtab->table[i]);
	}
	free(hashtab->table);
	free(hashtab);
}

static uint32_t _strhash(uint8_t *s, size_t hashsize) {
	uint32_t hashval;
	for (hashval = 0; *s != '\0'; ++s) {
		hashval = *s + 31 * hashval;
	}
	return hashval % hashsize;
}

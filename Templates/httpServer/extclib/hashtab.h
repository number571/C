#ifndef EXTCLIB_HASHTAB_H_
#define EXTCLIB_HASHTAB_H_

#include <stddef.h>
#include <stdint.h>

#include "type.h"

typedef struct HashTab HashTab;

extern HashTab *new_hashtab(size_t size, vtype_t key, vtype_t value);
extern void free_hashtab(HashTab *hashtab);

extern value_t get_hashtab(HashTab *hashtab, void *key);
extern int8_t set_hashtab(HashTab *hashtab, void *key, void *value);
extern void del_hashtab(HashTab *hashtab, void *key);
extern _Bool in_hashtab(HashTab *hashtab, void *key);

extern _Bool eq_hashtab(HashTab *x, HashTab *y);
extern size_t size_hashtab(HashTab *hashtab);
extern size_t sizeof_hashtab(void);

extern void print_hashtab(HashTab *hashtab);
extern void println_hashtab(HashTab *hashtab);

extern void print_hashtab_format(HashTab *hashtab);
extern void println_hashtab_format(HashTab *hashtab);

#endif /* EXTCLIB_HASHTAB_H_ */

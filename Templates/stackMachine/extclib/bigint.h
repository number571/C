#ifndef EXTCLIB_BIGINT_H_
#define EXTCLIB_BIGINT_H_

#include <stdio.h>
#include <stdint.h>

typedef struct BigInt BigInt;

extern int8_t set_base_bigint(uint8_t base);

extern BigInt *new_bigint(char *str);
extern void free_bigint(BigInt *x);

extern void in_bigint(BigInt *z, FILE *stream);
extern void out_bigint(FILE *stream, BigInt *z);

extern void inc_bigint(BigInt *z, BigInt *x);
extern void dec_bigint(BigInt *z, BigInt *x);

extern void mul_bigint(BigInt *z, BigInt *x, BigInt *y);
extern void div_bigint(BigInt *z, BigInt *x, BigInt *y);
extern void mod_bigint(BigInt *z, BigInt *x, BigInt *y);

extern void neg_bigint(BigInt *z, BigInt *x);
extern void abs_bigint(BigInt *z, BigInt *x);

extern void fact_bigint(BigInt *z, uint32_t x);
extern _Bool isprime_bigint(BigInt *z);
extern void gcdext_bigint(BigInt *z, BigInt *a, BigInt *b, BigInt *x, BigInt *y);
extern void gcd_bigint(BigInt *z, BigInt *a, BigInt *b);

extern void exp_bigint(BigInt *z, BigInt *x, size_t q);
extern void expmod_bigint(BigInt *z, BigInt *x, BigInt *e, BigInt *m);
extern void divmod_bigint(BigInt *q, BigInt *r, BigInt *x, BigInt *y);

extern void add_bigint(BigInt *z, BigInt *x, BigInt *y);
extern void sub_bigint(BigInt *z, BigInt *x, BigInt *y);
extern void inv_bigint(BigInt *z, BigInt *a, BigInt *b);

extern void shl_bigint(BigInt *z, BigInt *x, size_t q);
extern void shr_bigint(BigInt *z, BigInt *x, size_t q);

extern void xor_bigint(BigInt *z, BigInt *x, BigInt *y);
extern void and_bigint(BigInt *z, BigInt *x, BigInt *y);
extern void or_bigint(BigInt *z, BigInt *x, BigInt *y);
extern void not_bigint(BigInt *z, BigInt *x);

extern BigInt *dup_bigint(BigInt *x);
extern size_t size_bigint(BigInt *x);
extern size_t sizeof_bigint(void);

extern void cpy_bigint(BigInt *x, BigInt *y);
extern int8_t cmp_bigint(BigInt *x, BigInt *y);
extern _Bool eq_bigint(BigInt *x, BigInt *y);

extern void print_bigint(BigInt *x);
extern void println_bigint(BigInt *x);

extern void cpystr_bigint(BigInt *x, char *str);
extern void cpynum_bigint(BigInt *x, uint32_t num);

extern void getstr_bigint(char *str, BigInt *x);
extern uint32_t getnum_bigint(BigInt *x);

extern int8_t cmpnum_bigint(BigInt *x, uint32_t num);
extern _Bool eqnum_bigint(BigInt *x, uint32_t y);

extern void addnum_bigint(BigInt *z, BigInt *x, uint32_t num);
extern void subnum_bigint(BigInt *z, BigInt *x, uint32_t num);
extern void mulnum_bigint(BigInt *z, BigInt *x, uint32_t num);
extern void divnum_bigint(BigInt *z, BigInt *x, uint32_t num);
extern void modnum_bigint(BigInt *z, BigInt *x, uint32_t num);

#endif /* EXTCLIB_BIGINT_H_ */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <gmp.h>

#include "bigint.h"

typedef struct BigInt{
    mpz_t decimal;
} BigInt;

static enum Base {
    BIN = 2,
    OCT = 8,
    DEC = 10,
    HEX = 16,
} mode = DEC;

extern int8_t set_base_bigint(uint8_t base) {
    switch(base) {
        case 2: case 8: case 10: case 16:
            mode = (enum Base)base;
        break;
        default: return 1;
    }
    return 0;
}

extern BigInt *new_bigint(char *str) {
    BigInt *bignum = (BigInt*)malloc(sizeof(BigInt));
    mpz_init(bignum->decimal);
    mpz_set_str(bignum->decimal, str, mode);
    return bignum;
}

extern void free_bigint(BigInt *x) {
    mpz_clear(x->decimal);
    free(x);
}

extern void in_bigint(BigInt *z, FILE *stream) {
    mpz_inp_str(z->decimal, stream, mode);
}

extern void out_bigint(FILE *stream, BigInt *z) {
    mpz_out_str(stream, mode, z->decimal);
}

extern void inc_bigint(BigInt *z, BigInt *x) {
    mpz_add_ui(z->decimal, x->decimal, 1);
}

extern void dec_bigint(BigInt *z, BigInt *x) {
    mpz_sub_ui(z->decimal, x->decimal, 1);
}

extern void mul_bigint(BigInt *z, BigInt *x, BigInt *y) {
    mpz_mul(z->decimal, x->decimal, y->decimal);
}

extern void div_bigint(BigInt *z, BigInt *x, BigInt *y) {
    mpz_div(z->decimal, x->decimal, y->decimal);
}

extern void mod_bigint(BigInt *z, BigInt *x, BigInt *y) {
    mpz_mod(z->decimal, x->decimal, y->decimal);
}

extern void neg_bigint(BigInt *z, BigInt *x) {
    mpz_neg(z->decimal, x->decimal);
}

extern void abs_bigint(BigInt *z, BigInt *x) {
    mpz_abs(z->decimal, x->decimal);
}

extern void fact_bigint(BigInt *z, uint32_t x) {
    mpz_fac_ui(z->decimal, x);
}

extern _Bool isprime_bigint(BigInt *z) {
    const size_t q = 25;
    return mpz_probab_prime_p(z->decimal, q);
}

extern void gcdext_bigint(BigInt *z, BigInt *a, BigInt *b, BigInt *x, BigInt *y) {
    mpz_gcdext(z->decimal, x->decimal, y->decimal, a->decimal, b->decimal);
}

extern void gcd_bigint(BigInt *z, BigInt *a, BigInt *b) {
    mpz_gcd(z->decimal, a->decimal, b->decimal);
}

extern void exp_bigint(BigInt *z, BigInt *x, size_t q) {
    mpz_pow_ui(z->decimal, x->decimal, q);
}

extern void expmod_bigint(BigInt *z, BigInt *x, BigInt *e, BigInt *m) {
    mpz_powm(z->decimal, x->decimal, e->decimal, m->decimal);
}

extern void divmod_bigint(BigInt *q, BigInt *r, BigInt *x, BigInt *y) {
    mpz_divmod(q->decimal, r->decimal, x->decimal, y->decimal);
}

extern void add_bigint(BigInt *z, BigInt *x, BigInt *y) {
    mpz_add(z->decimal, x->decimal, y->decimal);
}

extern void sub_bigint(BigInt *z, BigInt *x, BigInt *y) {
    mpz_sub(z->decimal, x->decimal, y->decimal);
}

extern void inv_bigint(BigInt *z, BigInt *x, BigInt *y) {
    mpz_invert(z->decimal, x->decimal, y->decimal);
}

extern void shl_bigint(BigInt *z, BigInt *x, size_t q) {
    mpz_mul_2exp(z->decimal, x->decimal, q);
}

extern void shr_bigint(BigInt *z, BigInt *x, size_t q) {
    mpz_div_2exp(z->decimal, x->decimal, q);
}

extern void xor_bigint(BigInt *z, BigInt *x, BigInt *y) {
    mpz_xor(z->decimal, x->decimal, y->decimal);
}

extern void and_bigint(BigInt *z, BigInt *x, BigInt *y) {
    mpz_and(z->decimal, x->decimal, y->decimal);
}

extern void or_bigint(BigInt *z, BigInt *x, BigInt *y) {
    mpz_ior(z->decimal, x->decimal, y->decimal);
}

extern void not_bigint(BigInt *z, BigInt *x) {
    mpz_com(z->decimal, x->decimal);
}

extern void cpy_bigint(BigInt *x, BigInt *y) {
    mpz_set(x->decimal, y->decimal);
}

extern void cpynum_bigint(BigInt *x, uint32_t num) {
    mpz_set_ui(x->decimal, num);
}

extern void cpystr_bigint(BigInt *x, char *str) {
    mpz_set_str(x->decimal, str, mode);
}

extern BigInt *dup_bigint(BigInt *x) {
    BigInt *bignum = (BigInt*)malloc(sizeof(BigInt));
    mpz_init(bignum->decimal);
    mpz_set(bignum->decimal, x->decimal);
    return bignum;
}

extern void getstr_bigint(char *str, BigInt *x) {
    mpz_get_str(str, mode, x->decimal);
}

extern uint32_t getnum_bigint(BigInt *x) {
    return mpz_get_ui(x->decimal);
}

extern size_t size_bigint(BigInt *x) {
    return mpz_sizeinbase(x->decimal, mode);
}

extern size_t sizeof_bigint(void) {
    return sizeof(BigInt);
}

extern _Bool eq_bigint(BigInt *x, BigInt *y) {
    return mpz_cmp(x->decimal, y->decimal) == 0;
}

extern _Bool eqnum_bigint(BigInt *x, uint32_t y) {
    return mpz_cmp_ui(x->decimal, y) == 0;
}

extern void addnum_bigint(BigInt *z, BigInt *x, uint32_t num) {
    mpz_add_ui(z->decimal, x->decimal, num);
}

extern void subnum_bigint(BigInt *z, BigInt *x, uint32_t num) {
    mpz_sub_ui(z->decimal, x->decimal, num);
}

extern void mulnum_bigint(BigInt *z, BigInt *x, uint32_t num) {
    mpz_mul_ui(z->decimal, x->decimal, num);
}

extern void divnum_bigint(BigInt *z, BigInt *x, uint32_t num) {
    mpz_div_ui(z->decimal, x->decimal, num);
}

extern void modnum_bigint(BigInt *z, BigInt *x, uint32_t num) {
    mpz_mod_ui(z->decimal, x->decimal, num);
}

extern int8_t cmp_bigint(BigInt *x, BigInt *y) {
    int cond = mpz_cmp(x->decimal, y->decimal);
    if (cond > 0) {
        return 1;
    } else if (cond < 0) {
        return -1;
    }
    return 0;
}

extern int8_t cmpnum_bigint(BigInt *x, uint32_t y) {
    int cond = mpz_cmp_ui(x->decimal, y);
    if (cond > 0) {
        return 1;
    } else if (cond < 0) {
        return -1;
    }
    return 0;
}

extern void print_bigint(BigInt *x) {
    mpz_out_str(stdout, mode, x->decimal);
}

extern void println_bigint(BigInt *x) {
    mpz_out_str(stdout, mode, x->decimal);
    putchar('\n');
}

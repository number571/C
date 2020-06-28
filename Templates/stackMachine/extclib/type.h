#ifndef EXTCLIB_TYPE_H_
#define EXTCLIB_TYPE_H_

#include <stdint.h>
#include "bigint.h"

typedef enum vtype_t {
    DECIMAL_TYPE,
    REAL_TYPE,
    STRING_TYPE,
    BIGINT_TYPE,
} vtype_t;

typedef union value_t {
    int32_t decimal;
    double real;
    char *string;
    BigInt *bigint;
} value_t;

extern void *decimal(int32_t x);
extern void *string(char *x);
extern void *real(double x);

#endif /* EXTCLIB_TYPE_H_ */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define SIZE   10
#define MODULO 1000000000
#define MAX(x, y) ((x > y) ? (x) : (y))

typedef struct {
	int8_t   sign;
	uint32_t *number;
	uint32_t size;
	uint32_t len;
} BigInt;

BigInt *new_bigint(uint64_t x);

BigInt *mod_bigint(BigInt *x, BigInt *y);
BigInt *div_bigint(BigInt *x, BigInt *y);
BigInt *mul_bigint(BigInt *x, BigInt *y);
BigInt *sub_bigint(BigInt *x, BigInt *y);
BigInt *add_bigint(BigInt *x, BigInt *y);
BigInt *cpy_bigint(BigInt *x, BigInt *y);
BigInt *xchg_bigint(BigInt *x, BigInt *y);
BigInt *neg_bigint(BigInt *x);

int8_t cmps_bigint(BigInt *x, BigInt *y);
int8_t cmp_bigint(BigInt *x, BigInt *y);

void print_bigint(BigInt *x);
void free_bigint(BigInt *x);

int main(void) {
	BigInt *x = new_bigint(999999999999999999);
	BigInt *y = new_bigint(9999999);

	mul_bigint(x, y); // x = x * y
	cpy_bigint(y, x); // y = x
	add_bigint(x, y); // x = x + y
	neg_bigint(x);    // x = -x

	print_bigint(x);

	free_bigint(x);
	free_bigint(y);
	return 0;
}

BigInt *neg_bigint(BigInt *x) {
	x->sign = -x->sign;
	return x;
}

void print_bigint(BigInt *x) {
	_Bool number_is_null = 1;
	putchar(x->sign == 1 ? '+' : '-');
	for (ssize_t i = x->size-1; i != -1; --i) {
		if (x->number[i] == 0 && number_is_null) {
			continue;
		}
		number_is_null = 0;
		printf("%.9u", x->number[i]);
	}
	if (number_is_null) {
		printf("0");
	}
	putchar('\n');
}

BigInt *cpy_bigint(BigInt *x, BigInt *y) {
	x->sign = y->sign;
	for (size_t i = 0; i < x->size; ++i) {
		x->number[i] = y->number[i];
	}
	return x;
}

BigInt *mod_bigint(BigInt *x, BigInt *y) {
	BigInt *null = new_bigint(0);
	int8_t code = cmp_bigint(y, null);

	if (code == 0 || code == -1) {
		return x;
	}

	BigInt *count = new_bigint(0);

	while ((code = cmp_bigint(x, y)) == 1 || code == 0) {
		sub_bigint(x, y);
	}

	free_bigint(count);
	free_bigint(null);
	return x;
}

BigInt *div_bigint(BigInt *x, BigInt *y) {
	BigInt *null = new_bigint(0);
	int8_t code = cmp_bigint(y, null);

	if (code == 0 || code == -1) {
		return x;
	}

	BigInt *one  = new_bigint(1);
	BigInt *count = new_bigint(0);

	while ((code = cmp_bigint(x, y)) == 1 || code == 0) {
		sub_bigint(x, y);
		add_bigint(count, one);
	}

	xchg_bigint(x, count);

	free_bigint(count);
	free_bigint(null);
	free_bigint(one);

	return x;
}

BigInt *mul_bigint(BigInt *x, BigInt *y) {
	BigInt *null = new_bigint(0);
	int8_t code = cmp_bigint(y, null);

	if (code == 0 || code == -1) {
		return x;
	}

	BigInt *one  = new_bigint(1);
	BigInt *cpx  = new_bigint(1);

	cpy_bigint(cpx, x);
	sub_bigint(y, one);

	while(cmp_bigint(y, null) != 0) {
		add_bigint(x, cpx);
		sub_bigint(y, one);
	}

	free_bigint(null);
	free_bigint(cpx);
	free_bigint(one);
	return x;
}

int8_t cmps_bigint(BigInt *x, BigInt *y) {
	if (x->sign == -1 && y->sign == 1) {
		return -1;
	}
	if (x->sign == 1 && y->sign == -1) {
		return 1;
	}
	if (x->sign == -1 && y->sign == -1) {
		return -cmp_bigint(x, y);
	}
	return cmp_bigint(x, y);
}

int8_t cmp_bigint(BigInt *x, BigInt *y) {
	for (ssize_t i = x->size-1; i != -1 ; --i) {
		if (x->number[i] > y->number[i]) {
			return 1;
		}
		if (x->number[i] < y->number[i]) {
			return -1;
		}
	}
	return 0;
}

BigInt *xchg_bigint(BigInt *x, BigInt *y) {
	BigInt *temp = new_bigint(1);
	cpy_bigint(temp, x);
	cpy_bigint(x, y);
	cpy_bigint(y, temp);
	free_bigint(temp);
	return x;
}

BigInt *sub_bigint(BigInt *x, BigInt *y) {
	if (x->sign == -1 && y->sign == 1) {
		x->sign = 1;
		add_bigint(x, y);
		x->sign = -1;
		return x;
	}

	if (x->sign == 1 && y->sign == -1) {
		y->sign = 1;
		add_bigint(x, y);
		return x;
	}

	if (cmp_bigint(x, y) == -1) {
		xchg_bigint(x, y);
		sub_bigint(x, y);
		neg_bigint(x);
		return x;
	}

	uint32_t length = MAX(x->len, y->len);
	uint32_t carry_flag = 0;
	uint32_t carry = 0;

	for (size_t i = 0; i < length; ++i) {
		uint64_t temp = (carry + MODULO + x->number[i]) - ((y->number[i] + carry_flag) % MODULO);
		if (temp >= MODULO) {
			temp = temp % MODULO;
			carry += temp / MODULO;
			carry_flag = 0;
		} else {
			carry_flag = 1;
		}
		x->number[i] = temp % MODULO;
		carry = temp / MODULO;
	}

	if (carry) {
		x->number[x->len++] = carry;
		if (x->len == x->size) {
			x->size *= 2;
			x->number = (uint32_t*)realloc(x->number, x->size * sizeof(uint32_t));
			memset(x->number + x->len, 0, (x->size - x->len) * sizeof(uint32_t));
		}
	}
	return x;
}

BigInt *add_bigint(BigInt *x, BigInt *y) {
	if (x->sign == -1 && y->sign == 1) {
		int8_t sign = 1;
		if (cmp_bigint(x, y) == 1) {
			sign = -1;
		}
		x->sign = 1;
		sub_bigint(x, y);
		x->sign = sign;
		return x;
	}

	if (x->sign == 1 && y->sign == -1) {
		y->sign = 1;
		sub_bigint(x, y);
		return x;
	}

	uint32_t length = MAX(x->len, y->len);
	uint32_t carry = 0;
	for (size_t i = 0; i < length; ++i) {
		uint64_t temp = carry + x->number[i] + y->number[i];
		x->number[i] = temp % MODULO;
		carry = temp / MODULO;
	}
	if (carry) {
		x->number[x->len++] = carry;
		if (x->len == x->size) {
			x->size *= 2;
			x->number = (uint32_t*)realloc(x->number, x->size * sizeof(uint32_t));
			memset(x->number + x->len, 0, (x->size - x->len) * sizeof(uint32_t));
		}
	}
	return x;
}

BigInt *new_bigint(uint64_t x) {
	BigInt *result = (BigInt*)malloc(sizeof(BigInt));
	result->sign   = 1;
	result->size   = SIZE;
	result->len    = 2;
	result->number = (uint32_t*)malloc(result->size * sizeof(uint32_t));
	memset(result->number, 0, result->size * sizeof(uint32_t));

	result->number[0] = (uint32_t)(x % MODULO);
	result->number[1] = (uint32_t)(x / MODULO);

	uint32_t carry = 0;
	for (size_t i = 0; i < result->len; ++i) {
		uint64_t temp = carry + result->number[i];
		result->number[i] = temp % MODULO;
		carry = temp / MODULO;
	}

	if (carry) {
		result->number[result->len++] = carry;
	}

	return result;
}

void free_bigint(BigInt *x) {
	free(x->number);
	free(x);
}

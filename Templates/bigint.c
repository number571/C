#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MODULO 1000000000

// // count = 6
// // size = 6 * 2 = 12 * 2 = 24 ...
// [ 11, 22, 33, 00, 00, 00 ]
// 0 - null

// * 2
// << 1

typedef struct BigInt {
	uint32_t *number;
	size_t count;
	size_t size;
} BigInt;

static void _swap(uint8_t *array, size_t size) {
	uint8_t temp;
	for (size_t i = 0, j = size-1; i < size / 2; ++i, --j) {
		temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
}

static void _inc_count_bigint(BigInt *x) {
	x->count += 1;
	if (x->count == x->size) {
		x->size <<= 1;
		x->number = (uint32_t*)realloc(x->number, x->size * sizeof(uint32_t));
		memset(x->number + x->count, 0, (x->size - x->count) * sizeof(uint32_t));
	}
}

static void _set_count_bigint(BigInt *x, BigInt *y) {
	if (x->size < y->size) {
		x->size = y->size;
		x->number = (uint32_t*)realloc(x->number, x->size * sizeof(uint32_t));
		memset(x->number + x->count, 0, (x->size - x->count) * sizeof(uint32_t));
	}
	if (y->size < x->size) {
		y->size = x->size;
		y->number = (uint32_t*)realloc(y->number, y->size * sizeof(uint32_t));
		memset(y->number + y->count, 0, (y->size - y->count) * sizeof(uint32_t));
	}
	if (x->count < y->count) {
		x->count = y->count;
	}
}

static void _clr_count_bigint(BigInt *x) {
	if (x->count == 0) {
		return; // null
	}
	size_t i = x->count-1;
	for (; i > 0; --i) {
		if (x->number[i] != 0) {
			break;
		}
	}
	x->count = i + 1;
}

extern BigInt *new_bigint(uint8_t *str);
extern void free_bigint(BigInt *x);

extern void add_bigint(BigInt *x, BigInt *y);
extern void sub_bigint(BigInt *x, BigInt *y);

extern int8_t cmp_bigint(BigInt *x, BigInt *y);
extern void xchg_bigint(BigInt *x, BigInt *y);

extern void print_bigint(BigInt *x);

int main(void) {
	BigInt *x = new_bigint("0");
	BigInt *y = new_bigint("1");

	sub_bigint(x, y);
	print_bigint(x);

	free_bigint(x);
	free_bigint(y);
	return 0;
}

extern void xchg_bigint(BigInt *x, BigInt *y) {
	BigInt temp = *x;
	*x = *y;
	*y = temp;
}

extern int8_t cmp_bigint(BigInt *x, BigInt *y) {
	_set_count_bigint(x, y);
	for (ssize_t i = x->count-1; i != -1; --i) {
		if (x->number[i] < y->number[i]) {
			return -1;
		}
		if (x->number[i] > y->number[i]) {
			return 1;
		}
	}
	return 0;
}

extern void print_bigint(BigInt *x) {
	if (x->count == 0) {
		printf("null\n");
		return;
	}
	ssize_t i = x->count-1;
	for (; i > 0; --i) {
		if (x->number[i] != 0) {
			break;
		}
	}
	printf("%u", x->number[i]);
	for (--i; i != -1; --i) {
		printf("%.9u", x->number[i]);
	}
	putchar('\n');
}

extern void add_bigint(BigInt *x, BigInt *y) {
	_set_count_bigint(x, y);
	uint8_t carry = 0;
	for (size_t i = 0; i < x->count; ++i) {
		uint64_t temp = carry + x->number[i] + y->number[i];
		x->number[i] = temp % MODULO;
		carry = temp / MODULO;
	}
	if (carry) {
		x->number[x->count] = carry;
		_inc_count_bigint(x);
	}
	_clr_count_bigint(x);
}

extern void sub_bigint(BigInt *x, BigInt *y) {
	if (cmp_bigint(x, y) == -1) {
		xchg_bigint(x, y);
		sub_bigint(x, y);
		return;
	}
	_set_count_bigint(x, y);
	uint8_t carry = 0;
	for (size_t i = 0; i < x->count; ++i) {
		uint64_t temp = (MODULO + x->number[i]) - (carry + y->number[i]);
		x->number[i] = temp % MODULO;
		carry = (temp >= MODULO) ? 0 : 1;
	}
	_clr_count_bigint(x);
}

extern BigInt *new_bigint(uint8_t *str) {
	const size_t BUFFSIZE = 9;

	BigInt *bigint = (BigInt*)malloc(sizeof(BigInt));
	bigint->size   = 10;
	bigint->count  = 0;
	bigint->number = (uint32_t*)malloc(bigint->size * sizeof(uint32_t));

	uint8_t chunck[BUFFSIZE+1];
	memset(chunck, 0, (BUFFSIZE+1) * sizeof(uint8_t));

	size_t next = 0;
	size_t index = 0;
	size_t length = strlen(str);

	for (ssize_t i = length-1; i != -1; --i) {
		chunck[index++] = str[i];
		if (index == BUFFSIZE) {
			index = 0;
			_swap(chunck, BUFFSIZE);
			bigint->number[next++] = atoi(chunck);
			_inc_count_bigint(bigint);	
		}
	}
	if (index != 0) {
		chunck[index] = '\0';
		_swap(chunck, index);
		bigint->number[next++] = atoi(chunck);
		_inc_count_bigint(bigint);
	}

	return bigint;
}

extern void free_bigint(BigInt *x) {
	free(x->number);
	free(x);
}

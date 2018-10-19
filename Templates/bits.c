#include <stdio.h>

typedef union {
	unsigned char byte;
	struct {
		unsigned char _0: 1;
		unsigned char _1: 1;
		unsigned char _2: 1;
		unsigned char _3: 1;
		unsigned char _4: 1;
		unsigned char _5: 1;
		unsigned char _6: 1;
		unsigned char _7: 1;
	} bit;
} Byte;

void print_bits (Byte x) {
	printf("%hhu%hhu%hhu%hhu%hhu%hhu%hhu%hhu",
		x.bit._7, x.bit._6, x.bit._5, x.bit._4,
		x.bit._3, x.bit._2, x.bit._1, x.bit._0);
}

int main(void) {
	Byte x = {.byte = 5};
	Byte y = {.byte = 2};
	Byte z = {.byte = x.byte << y.byte};

	print_bits(x); printf(" [%hhu]\n", x.byte);
	print_bits(y); printf(" [%hhu]\n", y.byte);
	print_bits(z); printf(" [%hhu]\n", z.byte);

    return 0;
}

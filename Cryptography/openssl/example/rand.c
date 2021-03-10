#include "extclib/crypto.h"

#include <stdio.h>
#include <string.h>

static void print_bytes(char *hash, size_t size);

int main(void) {
	char randbytes[32];

	crypto_rand(randbytes, 32);
	print_bytes(randbytes, 32);

	return 0;
}

static void print_bytes(char *hash, size_t size) {
	printf("[ ");
	for (size_t i = 0; i < size; ++i) {
		printf("%d ", hash[i]);
	}
	printf("]\n");
}

#include "extclib/crypto.h"

#include <stdio.h>
#include <string.h>

static void print_bytes(char *hash, size_t size);

int main(void) {
	char hash[32];
	char message[] = "hello, world!";

	crypto_sha_256(hash, message, strlen(message));
	print_bytes(hash, 32);

	return 0;
}

static void print_bytes(char *hash, size_t size) {
	printf("[ ");
	for (size_t i = 0; i < size; ++i) {
		printf("%d ", hash[i]);
	}
	printf("]\n");
}

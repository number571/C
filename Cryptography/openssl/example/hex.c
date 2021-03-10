#include "extclib/crypto.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char const *argv[]) {
	char encoded[100];
	char decoded[10];
    char bytes[10] = { 1, -5, 20, -11, 122, 3, 6, 0, 19, -120};

    crypto_hex(1, encoded, 100, bytes, 10);
    printf("%s\n", encoded);

	crypto_hex(-1, decoded, 10, encoded, strlen(encoded));
    for (int i = 0; i < 10; ++i) {
    	printf("%d ", decoded[i]);
    }
    printf("\n");

    return 0;
}

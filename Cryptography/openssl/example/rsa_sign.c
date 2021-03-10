#include "extclib/crypto.h"

#include <stdio.h>
#include <string.h>

int main(void) {
	crypto_rsa *key = crypto_rsa_new(2048);
	char buffer[BUFSIZ];

	char message[] = "hello, world!";
	crypto_rsa_sign(1, key, buffer, BUFSIZ, message, strlen(message));

	// message[5] = 'a';

	if (crypto_rsa_sign(-1, key, buffer, BUFSIZ, message, strlen(message)) == 0) {
		printf("verify success\n");
	} else {
		printf("verify failed\n");
	}

	crypto_rsa_free(key);
	return 0;
}

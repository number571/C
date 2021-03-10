#include "extclib/crypto.h"

#include <stdio.h>
#include <string.h>

int main(void) {
	crypto_rsa *key = crypto_rsa_new(2048);
	char buffer[BUFSIZ];

	char message[] = "hello, world!";
	crypto_rsa_oaep(1, key, buffer, BUFSIZ, message, strlen(message)+1);

	crypto_rsa_oaep(-1, key, buffer, BUFSIZ, buffer, crypto_rsa_size(key));
	printf("%s\n", buffer);

	crypto_rsa_free(key);
	return 0;
}

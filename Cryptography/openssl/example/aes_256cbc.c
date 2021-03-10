#include "extclib/crypto.h"

#include <stdio.h>
#include <string.h>

int main(void) {
	char exkey[32];
	char iv[16];
	char buffer[BUFSIZ];

	char message[128] = "Yet better thus, and known to be contemn'd, Than still contemn'd and flatter'd.";
	char key[] = "hello, world!";

	crypto_rand(iv, 16);
	crypto_sha_256(exkey, key, strlen(key));
	int n = crypto_aes_256cbc(1, exkey, buffer, message, strlen(message), iv);

	crypto_aes_256cbc(-1, exkey, buffer, buffer, n, iv);
	printf("[%d] %s\n", n, buffer);

	return 0;
}

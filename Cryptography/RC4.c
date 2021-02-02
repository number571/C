#include <stdio.h>
#include <stdint.h>
#include <string.h>

static uint8_t Sbox[256];

static void swap(uint8_t *x, uint8_t *y) {
	uint8_t t = 0;
	t = *x;
	*x = *y;
	*y = t;
}

extern void rc4_init(char *key, int ksize) {
	uint8_t j = 0;
	for (int i = 0; i < 256; ++i) {
		Sbox[i] = i;
	}
	for (int i = 0; i < 256; ++i) {
		j = j + Sbox[i] + (uint8_t)key[i % ksize];
		swap(&Sbox[i], &Sbox[j]);
	}
}

extern void rc4_generate(char *output, int size) {
	uint8_t i = 0, j = 0;
	uint8_t t;
	int k = 0;
	for (int k = 0; k < size; ++k) {
		i += 1;
		j += Sbox[i];
		swap(&Sbox[i], &Sbox[j]);
		t = Sbox[i] + Sbox[j];
		output[k] = (char)Sbox[t];
	}
}

extern void xor_encrypt(char *output, char *key, int ksize, char *input, int size) {
	for (int i = 0; i < size; ++i) {
		output[i] = input[i] ^ key[i % ksize];
	}
}

static void print_bytes(char *bytes, int size) {
	for (int i = 0; i < size; ++i) {
		printf("%d ", (uint8_t)bytes[i]);
	}
	printf("\n");
}

int main(void) {
	char *key = "it's a key!";
	int klen = strlen(key);
	char kbytes[BUFSIZ];

	rc4_init(key, klen);
	rc4_generate(kbytes, BUFSIZ);
	
	char output[BUFSIZ] = "hello, world!";
	int mlen = strlen(output);
	print_bytes(output, mlen);

	xor_encrypt(output, kbytes, BUFSIZ, output, mlen);
	print_bytes(output, mlen);

	xor_encrypt(output, kbytes, BUFSIZ, output, mlen);
	print_bytes(output, mlen);

	return 0;
}

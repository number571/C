#include <stdio.h>
#include <stdint.h>

#define BUFF_SIZE 1024

size_t TEA(uint8_t * to, uint8_t mode, uint8_t * key128b, uint8_t * from, size_t length);
void feistel_cipher(uint8_t mode, uint32_t * block32b_1, uint32_t * block32b_2, uint32_t * keys32b);

void split_128bits_to_32bits(uint8_t * key128b, uint32_t * keys32b);
void split_64bits_to_32bits(uint64_t block64b, uint32_t * block32b_1, uint32_t * block32b_2);
void split_64bits_to_8bits(uint64_t block64b, uint8_t * blocks8b);

uint64_t join_32bits_to_64bits(uint32_t block32b_1, uint32_t block32b_2);
uint64_t join_8bits_to_64bits(uint8_t * blocks8b);

static inline size_t input_string(uint8_t * buffer);
static inline void print_array(uint8_t * array, size_t length);
static inline void print_bits(uint64_t x, register uint64_t Nbit);

int main(void) {
	uint8_t encrypted[BUFF_SIZE], decrypted[BUFF_SIZE];
	uint8_t buffer[BUFF_SIZE];
	uint8_t key128b[16] = "TEA_password_key";

	size_t length = input_string(buffer);
	print_array(buffer, length);

	length = TEA(encrypted, 'E', key128b, buffer, length);
	print_array(encrypted, length);

	length = TEA(decrypted, 'D', key128b, encrypted, length);
	print_array(decrypted, length);

	return 0;
}

size_t TEA(uint8_t * to, uint8_t mode, uint8_t * key128b, uint8_t * from, size_t length) {
	length = length % 8 == 0 ? length : length + (8 - (length % 8));

	uint32_t N1, N2, keys32b[4];
	split_128bits_to_32bits(key128b, keys32b);

	for (size_t i = 0; i < length; i += 8) {
		split_64bits_to_32bits(
			join_8bits_to_64bits(from + i), 
			&N1, &N2
		);
		feistel_cipher(mode, &N1, &N2, keys32b);
		split_64bits_to_8bits(
			join_32bits_to_64bits(N1, N2),
			(to + i)
		);
	}

	return length;
}

void feistel_cipher(uint8_t mode, uint32_t * N1, uint32_t * N2, uint32_t * keys32b) {
	// C = (sqrt(5) - 1) * 2 ^ 31
	const uint32_t C = 0x9E3779B9;
	switch(mode) {
		case 'E': case 'e': {
			uint32_t T = 0;
			for (uint8_t round = 0; round < 32; ++round) {
				T = (T + C) % UINT32_MAX;
				*N1 += ((*N2 << 4) + keys32b[0]) ^ (*N2 + T) ^ ((*N2 >> 5) + keys32b[1]);
        		*N2 += ((*N1 << 4) + keys32b[2]) ^ (*N1 + T) ^ ((*N1 >> 5) + keys32b[3]);
			}
			break;
		}
		case 'D': case 'd': {
			uint32_t T = 0xC6EF3720;
			for (uint8_t round = 0; round < 32; ++round) {
				*N2 -= ((*N1 << 4) + keys32b[2]) ^ (*N1 + T) ^ ((*N1 >> 5) + keys32b[3]);
        		*N1 -= ((*N2 << 4) + keys32b[0]) ^ (*N2 + T) ^ ((*N2 >> 5) + keys32b[1]);
        		T = (T - C) % UINT32_MAX;
			}
			break;
		}
	}
}

void split_128bits_to_32bits(uint8_t * key128b, uint32_t * keys32b) {
	uint8_t *p8 = key128b;
	for (uint32_t *p32 = keys32b; p32 < keys32b + 4; ++p32) {
		for (uint8_t i = 0; i < 4; ++i) {
			*p32 = (*p32 << 8) | *(p8 + i);
		}
		p8 += 4;
	}
}

void split_64bits_to_32bits(uint64_t block64b, uint32_t * block32b_1, uint32_t * block32b_2) {
	*block32b_1 = (uint32_t)(block64b >> 32);
	*block32b_2 = (uint32_t)(block64b);
}

void split_64bits_to_8bits(uint64_t block64b, uint8_t * blocks8b) {
	for (size_t i = 0; i < 8; ++i) {
		blocks8b[i] = (uint8_t)(block64b >> ((7 - i) * 8));
	}
}

uint64_t join_32bits_to_64bits(uint32_t block32b_1, uint32_t block32b_2) {
	uint64_t block64b;
	block64b = (uint64_t)block32b_1;
	block64b = (uint64_t)(block64b << 32) | block32b_2;
	return block64b;
}

uint64_t join_8bits_to_64bits(uint8_t * blocks8b) {
	uint64_t block64b;
	for (uint8_t *p = blocks8b; p < blocks8b + 8; ++p) {
		block64b = (block64b << 8) | *p;
	}
	return block64b;
}

static inline size_t input_string(uint8_t * buffer) {
	size_t position = 0;
	uint8_t ch;
	while ((ch = getchar()) != '\n' && position < BUFF_SIZE - 1)
		buffer[position++] = ch;
	buffer[position] = '\0';
	return position;
}

static inline void print_array(uint8_t * array, size_t length) {
	printf("[ ");
	for (size_t i = 0; i < length; ++i)
		printf("%d ", array[i]);
	printf("]\n");
}

static inline void print_bits(uint64_t x, register uint64_t Nbit) {
	for (Nbit = (uint64_t)1 << (Nbit - 1); Nbit > 0x00; Nbit >>= 1)
		printf("%d", (x & Nbit) ? 1 : 0);
	putchar('\n');
}


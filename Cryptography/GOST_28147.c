#include <stdio.h>
#include <stdint.h>

#define LSHIFT_nBIT(x, L, N) (((x << L) | (x >> (-L & (N - 1)))) & (((uint64_t)1 << N) - 1))
#define RSHIFT_nBIT(x, R, N) (((x >> R) | (x << (-R & (N - 1)))) & (((uint64_t)1 << N) - 1))

uint64_t cipher_GOST_28147(uint8_t mode, uint64_t block64bits, uint8_t * key256bits);
void feistel_cipher(uint8_t mode, uint32_t * N1, uint32_t * N2, uint32_t * keys32bits);
void round_of_feistel_cipher(uint32_t * N1, uint32_t * N2, uint32_t * keys32bits, uint8_t round);

uint32_t substitution_table(uint32_t N1, uint8_t sbox_row);
void substitution_table_by_4bits(uint8_t * blocks4bits, uint8_t sbox_row);

void split_256bits_to_32bits(uint8_t * key256bits, uint32_t * keys32bits);
void split_64bits_to_32bits(uint64_t block, uint32_t * N1, uint32_t * N2);
void split_32bits_to_4bits(uint32_t N1, uint8_t * blocks4bits);

uint64_t join_32bits_to_64bits(uint32_t N1, uint32_t N2);
uint32_t join_4bits_to_32bits(uint8_t * blocks4bits);
uint64_t join_8bits_to_64bits(uint8_t * vector);

static inline void printNbits(uint64_t x, register uint64_t Nbit);

static uint8_t Sbox[8][16] = { 
	{0xF, 0xC, 0x2, 0xA, 0x6, 0x4, 0x5, 0x0, 0x7, 0x9, 0xE, 0xD, 0x1, 0xB, 0x8, 0x3}, 
	{0xB, 0x6, 0x3, 0x4, 0xC, 0xF, 0xE, 0x2, 0x7, 0xD, 0x8, 0x0, 0x5, 0xA, 0x9, 0x1},
	{0x1, 0xC, 0xB, 0x0, 0xF, 0xE, 0x6, 0x5, 0xA, 0xD, 0x4, 0x8, 0x9, 0x3, 0x7, 0x2},
	{0x1, 0x5, 0xE, 0xC, 0xA, 0x7, 0x0, 0xD, 0x6, 0x2, 0xB, 0x4, 0x9, 0x3, 0xF, 0x8},
	{0x0, 0xC, 0x8, 0x9, 0xD, 0x2, 0xA, 0xB, 0x7, 0x3, 0x6, 0x5, 0x4, 0xE, 0xF, 0x1},
	{0x8, 0x0, 0xF, 0x3, 0x2, 0x5, 0xE, 0xB, 0x1, 0xA, 0x4, 0x7, 0xC, 0x9, 0xD, 0x6},
	{0x3, 0x0, 0x6, 0xF, 0x1, 0xE, 0x9, 0x2, 0xD, 0x8, 0xC, 0x4, 0xB, 0xA, 0x5, 0x7},
	{0x1, 0xA, 0x6, 0x8, 0xF, 0xB, 0x0, 0x4, 0xC, 0x3, 0x5, 0x9, 0x7, 0xD, 0x2, 0xE},
};

int main(void) {
	uint8_t block[8] = {65, 66, 67, 0, 0, 0, 0, 2};
	uint64_t block64bits = join_8bits_to_64bits(block);

	uint8_t key256bits[32] = {
		65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
		81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
	};

	uint64_t cipher = cipher_GOST_28147('E', block64bits, key256bits);
	printNbits(cipher, (uint64_t)INT64_MAX+1);

	uint64_t decrypted = cipher_GOST_28147('D', cipher, key256bits);
	printNbits(decrypted, (uint64_t)INT64_MAX+1);

	return 0;
}

uint64_t cipher_GOST_28147(uint8_t mode, uint64_t block64bits, uint8_t * key256bits) {
	uint32_t N1, N2, keys32bits[8];
	split_256bits_to_32bits(key256bits, keys32bits);
	split_64bits_to_32bits(block64bits, &N1, &N2);
	feistel_cipher(mode, &N1, &N2, keys32bits);
	return join_32bits_to_64bits(N1, N2);
}

uint64_t join_32bits_to_64bits(uint32_t N1, uint32_t N2) {
	uint64_t block64bits;
	block64bits = N1;
	block64bits = (block64bits << 32) | N2;
	return block64bits;
}

void split_256bits_to_32bits(uint8_t * key256bits, uint32_t * keys32bits) {
	uint8_t *p8 = key256bits;
	for (uint32_t *p32 = keys32bits; p32 < keys32bits + 8; ++p32) {
		for (uint8_t i = 0; i < 4; ++i)
			*p32 = (*p32 << 8) | *(p8 + i);
		p8 += 4;
	}
}

void feistel_cipher(uint8_t mode, uint32_t * N1, uint32_t * N2, uint32_t * keys32bits) {
	switch (mode) {
		case 'E': case 'e': {
			for (uint8_t round = 0; round < 24; ++round)
				round_of_feistel_cipher(N1, N2, keys32bits, round);

			for (uint8_t round = 31; round >= 24; --round)
				round_of_feistel_cipher(N1, N2, keys32bits, round);
			break;
		}
		case 'D': case 'd': {
			for (uint8_t round = 0; round < 8; ++round)
				round_of_feistel_cipher(N1, N2, keys32bits, round);

			for (uint8_t round = 31; round >= 8; --round)
				round_of_feistel_cipher(N1, N2, keys32bits, round);
			break;
		}
	}
}

void round_of_feistel_cipher(uint32_t * N1, uint32_t * N2, uint32_t * keys32bits, uint8_t round) {
	uint32_t result_of_iter, temp;

	result_of_iter = *N1 + keys32bits[round % 8] % UINT32_MAX;
	result_of_iter = substitution_table(result_of_iter, round % 8);
	result_of_iter = (uint32_t)LSHIFT_nBIT(result_of_iter, 11, 32);

	temp = *N1;
	*N1 = *N2 ^ result_of_iter;
	*N2 = temp;
}

uint32_t substitution_table(uint32_t N1, uint8_t sbox_row) {
	uint8_t blocks4bits[4];
	split_32bits_to_4bits(N1, blocks4bits);
	substitution_table_by_4bits(blocks4bits, sbox_row);
	return join_4bits_to_32bits(blocks4bits);
}

void split_32bits_to_4bits(uint32_t N1, uint8_t * blocks4bits) {
	for (uint8_t i = 0; i < 4; ++i) {
		blocks4bits[i] = (uint8_t)(N1 >> (28 - (i * 8)));
		blocks4bits[i] = (blocks4bits[i] << 4) | (uint8_t)(N1 >> (24 - (i * 8)));
	}
}

void substitution_table_by_4bits(uint8_t * blocks4bits, uint8_t sbox_row) {
	uint8_t x, y;
	for (uint8_t i = 0; i < 4; ++i) {
		x = Sbox[sbox_row][blocks4bits[i] & 0x0F];
		y = Sbox[sbox_row][blocks4bits[i] >> 4];
		blocks4bits[i] = y;
		blocks4bits[i] = (blocks4bits[i] << 4) | x;
	}
}

uint32_t join_4bits_to_32bits(uint8_t * blocks4bits) {
	uint32_t block32bits;
	block32bits = blocks4bits[0];
	for (uint8_t i = 0; i < 4; ++i)
		block32bits = (block32bits << 8) | blocks4bits[i];
}

void split_64bits_to_32bits(uint64_t block, uint32_t * N1, uint32_t * N2) {
	*N1 = (uint32_t)block;
	*N2 = (uint32_t)(block >> 32);
}

uint64_t join_8bits_to_64bits(uint8_t * vector) {
	uint64_t result = 0;
	for (uint8_t *p = vector; p < vector + 8; ++p)
		result = (result << 8) | *p;
	return result;
}

static inline void printNbits(uint64_t x, register uint64_t Nbit) {
	for (; Nbit > 0x00; Nbit >>= 1)
		printf("%d", (x & Nbit) ? 1 : 0);
	putchar('\n');
}

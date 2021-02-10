#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generate_matrix3x3(int matrix[3][3], int mod);
void print_matrix3x3(int matrix[3][3]);
int det_matrix3x3(int matrix[3][3]);

int main(void) {
	int matrix[3][3];
	int mod = 29;
	int det;
	generate_matrix3x3(matrix, mod);
	print_matrix3x3(matrix);
	det = det_matrix3x3(matrix);
	printf("det=%d, det(mod n=%d)=%d\n", det, mod, (det % mod + mod) % mod);
	return 0;
}

void generate_matrix3x3(int matrix[3][3], int mod) {
	srand(time(NULL));
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			matrix[i][j] = rand() % mod;
		}
	}
}

void print_matrix3x3(int matrix[3][3]) {
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			printf("%2d ", matrix[i][j]);
		}
		printf("\n");
	}
}

int det_matrix3x3(int m[3][3]) {
	return 	m[0][0] * m[1][1] * m[2][2] + 
			m[0][1] * m[1][2] * m[2][0] +
			m[1][0] * m[2][1] * m[0][2] -
			m[2][0] * m[1][1] * m[0][2] -
			m[1][0] * m[0][1] * m[2][2] - 
			m[0][0] * m[1][2] * m[2][1];
}

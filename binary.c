#include <stdio.h>
void bin(char num);

int main(void) {
	char number; scanf("%hhd",&number); bin(number);
	return 0;
}

void bin(char num) {
	for (short bit = 0x80; bit > 0; bit /= 2) {
		printf("%hhd",(num&bit)?1:0);
	} printf("\n");
}

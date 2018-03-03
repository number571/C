#include <stdio.h>
void bin(int num);
int main(void) {
	char number;
	scanf("%hhd",&number);
	bin(number);
	return 0;
}
void bin(int num) {
	for (short bit = 0x80; bit > 0; bit /= 2) {
		printf("%d",(num&bit)?1:0);
	} printf("\n");
}

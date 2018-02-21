#include <stdio.h>
void bin(int num);
int main(void) {
	unsigned char number = 5;
	bin(number);
	printf("%d\n", number);
	return 0;
}
void bin(int num) {
	printf("%d",  (num&0x80)?1:0); // 0x80 = 10000000 
	printf("%d",  (num&0x40)?1:0); // 0x40 = 01000000
	printf("%d",  (num&0x20)?1:0); // 0x20 = 00100000
	printf("%d",  (num&0x10)?1:0); // 0x10 = 00010000
	printf("%d",  (num&0x08)?1:0); // 0x08 = 00001000
	printf("%d",  (num&0x04)?1:0); // 0x04 = 00000100
	printf("%d",  (num&0x02)?1:0); // 0x02 = 00000010
	printf("%d\n",(num&0x01)?1:0); // 0x01 = 00000001
}
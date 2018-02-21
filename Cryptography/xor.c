#include <stdio.h>
#include <string.h>
#define BUFFER 128
char *encryptDecrypt(short key, char *message);
int main(void) {
	short key; scanf("%hd ", &key);
	char text[BUFFER], temp;
	for (unsigned short index = 0; index < BUFFER; index++)
		if ((temp = getchar()) != '\n')
			text[index] = temp;
		else break;
	printf("Final message: %s\n", encryptDecrypt(key, text));
	return 0;
}
char *encryptDecrypt(short key, char *message) {
	for (short index = 0; index < strlen(message); index ++) {
		message[index] = message[index] ^ key;
	}
	return message;
}

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#define BUFFER 128

char *encryptDecrypt(char mode, char *message, short key);

int main(void) {
	char mode; short key;
	scanf("%c %hd ", &mode, &key);

	char text[BUFFER], temp;
	for (unsigned short index = 0; index < BUFFER; index++)
		if ((temp = getchar()) != '\n')
			text[index] = temp;
		else break;

	mode = toupper(mode);
	for (short index = 0; index < strlen(text); index++)
		text[index] = toupper(text[index]);

	printf("Final message: %s\n", encryptDecrypt(mode, text, key));
	return 0;
}

char *encryptDecrypt(char mode, char *message, short key) {
	if (mode == 'E') {
		for (short index = 0; index < strlen(message); index++)
			message[index] = (*(message + index) + key - 13) % 26 + 'A';
	} else {
		for (short index = 0; index < strlen(message); index++)
			message[index] = (*(message + index) - key - 13) % 26 + 'A';
	}
	return message;
}
#include <stdio.h>
#include <string.h>

#define BUFF 1024

void reverseWords(const char *message, char buffer[]);

int main(void) {
	const char *message = "Hello World! What's up?";
	char buffer[BUFF];

	reverseWords(message, buffer);

	printf("%s\n", buffer);

	return 0;
}

void reverseWords(const char *message, char buffer[]) {

	const unsigned short length = strlen(message);
	unsigned short position = length, local = 0;

	for (unsigned short index = length-1; index > 0; index--)
		if (message[index] == ' ') {
			for (unsigned short twindex = index; twindex < position; twindex++)
				buffer[local++] = message[twindex];
			position = index;
		}

	buffer[local++] = ' ';

	for (unsigned short index = 0; index < position; index++) {
		buffer[local++] = message[index];
	}

	buffer[length+2] = '\0';

	return;
}

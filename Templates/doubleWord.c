#include <stdio.h>
#include <string.h>

#define BUFF 1024

void doubleWord(const char *message, unsigned short number, 
	char buffer[]);

int main(void) {
	const char *message = "Hello World! How are you?";
	char buffer[BUFF] = {0};

	doubleWord(message, 3, buffer);

	printf("%s\n", buffer);
	return 0;
}

void doubleWord(const char *message, unsigned short number, 
	char buffer[]) {

	number -= 1;
	const unsigned short length = strlen(message);
	unsigned short numWord = 0;

	char localBuffer[BUFF] = {0};
	unsigned short local = 0;

	for (unsigned short index = 0; index < length; index++) {
		if (message[index] == ' ' ||
			message[index] == '\t'||
			message[index] == '\n')
			numWord++;

		if (numWord < number)
			localBuffer[local++] = message[index];

		if (numWord == number)
			break;
	}

	strcat(buffer, localBuffer);

	numWord = 0;
	local = 0;

	for (unsigned short index = 0; index < length; index++) {
		if (message[index] == ' ' ||
			message[index] == '\t'||
			message[index] == '\n')
			numWord++;

		if (numWord == number)
			localBuffer[local++] = message[index];

		if (numWord > number)
			break;
	}

	localBuffer[local] = '\0';

	strcat(buffer, localBuffer);
	strcat(buffer, localBuffer);

	numWord = 0;
	local = 0;

	for (unsigned short index = 0; index < length; index++) {
		if (message[index] == ' ' ||
			message[index] == '\t'||
			message[index] == '\n')
			numWord++;

		if (numWord > number)
			localBuffer[local++] = message[index];
	}

	localBuffer[local] = '\0';
	strcat(buffer, localBuffer);

	return;
}

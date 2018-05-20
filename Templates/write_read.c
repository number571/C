#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef unsigned int uint32_t;

_Bool create(const char *filename);
_Bool input(const char *filename, const char *text);
char* output(const char *filename);

int main(void) {

	const char *name = "file.txt";
	const char *message = "Hello World!";

	printf("File: '%s'\n", name);
	printf("Text: '%s'\n", message);

	if (create(name) == 0)
		printf("[+] File created\n");
	else
		printf("[-] File not created\n");

	if (input(name, message) == 0)
		printf("[+] File << Text\n");
	else
		printf("[-] File x< Text\n");

	char *text = output(name);
	if (text != NULL)
		printf("[+] File >> '%s'\n", text);
	else
		printf("[-] File >x\n");

	free(text);
	return 0;
}

_Bool create(const char *filename) {
	FILE *file = fopen(filename, "w");
	if (file != NULL) 
		fclose(file);
	else return 1;
	return 0;
}

_Bool input(const char *filename, const char *text) {
	FILE *file = fopen(filename, "a");
	if (file != NULL) {
		uint32_t length = strlen(text);
		for (uint32_t index = 0; index < length; index++)
			putc(*(text + index), file);
		fclose(file);
	} else return 1;
	return 0;
}

char* output(const char *filename) {
	FILE *file = fopen(filename, "r");
	if (file != NULL) {
		
		fseek(file, 0, SEEK_END);
		uint32_t size = ftell(file);
		fseek(file, 0, SEEK_SET);

		char *content = (char*) malloc(size * sizeof(char));

		uint32_t index; char c;
		while ((c = getc(file)) != EOF)
			*(content + index++) = c;

		fclose(file);
		return content;
	} else return NULL;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEBUG_TURN

#define DEBUG_SUCCESS(x) printf("[+] SUCCESS: ["x"]\n")
#define DEBUG_FAILURE(x) printf("[x] FAILURE: ["x"]\n")

#define DEBUG_CREATE DEBUG_FAILURE("CREATE"):DEBUG_SUCCESS("CREATE")
#define DEBUG_INPUT DEBUG_FAILURE("INPUT"):DEBUG_SUCCESS("INPUT")
#define DEBUG_OUTPUT DEBUG_SUCCESS("OUTPUT"):DEBUG_FAILURE("OUTPUT")

typedef unsigned int uint32_t;

_Bool create(char *filename);
_Bool input(char *filename, char *text);
char* output(char *filename);

int main(void) {
	char *filename = "file.txt";
	char *message = "Hello World!";
	#ifdef DEBUG_TURN
		create(filename) ? DEBUG_CREATE;
		input(filename, message) ? DEBUG_INPUT;

		char *result = output(filename);
		result ? DEBUG_OUTPUT;
		printf("%s\n", result);
	#else
		create(filename);
		input(filename, message);

		char *result = output(filename);
		printf("%s\n", result);
	#endif

	return EXIT_SUCCESS;
}

_Bool create(char *filename) {
	FILE *file = fopen(filename, "w");
	if (file != NULL) {
		fclose(file);
		return EXIT_SUCCESS;
	} else return EXIT_FAILURE;
}

_Bool input(char *filename, char *text) {
	FILE *file = fopen(filename, "a");
	if (file != NULL) {

		uint32_t length = strlen(text);
		for (uint32_t index = 0; index < length; index++)
			putc(*(text+index), file);

		fclose(file);
		return EXIT_SUCCESS;
	} else return EXIT_FAILURE;
}

char* output(char *filename) {
	FILE *file = fopen(filename, "r");
	if (file != NULL) {
		fseek(file, 0, SEEK_END);
		uint32_t length = ftell(file);
		fseek(file, 0, SEEK_SET);

		uint32_t index = 0; char c;
		char *content = (char*) malloc(length * sizeof(char));

		while((c = getc(file)) != EOF)
			*(content + index++) = c;
		*(content + index) = '\0';

		fclose(file);
		return content;
	} else return NULL;
}

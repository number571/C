#include <stdio.h>
#include <stdlib.h>

struct person {
	char name[16];
	int age;
};

_Bool save(const char *filename, const struct person *ptr);
_Bool read(const char *filename);

int main(void) {
	const char *name = "file.dat";
	const struct person id_1 = {"Tom", 26};
	
	save(name, &id_1);
	read(name);
	return 0;
}

_Bool save(const char *filename, const struct person *ptr) {
	FILE *file = fopen(filename, "wb");
	if (file != NULL) {
		char *c = (char*)ptr;
		int size = sizeof(struct person);
		for (int index = 0; index < size; index++) {
			putc(*c++, file);
		}
		fclose(file);
	} else return 1;
	return 0;
}

_Bool read(const char *filename) {
	FILE *file = fopen(filename, "rb");
	if (file != NULL) {
		int size = sizeof(struct person) * sizeof(char);
		struct person *ptr = (struct person*) malloc(size);
		char *c = (char*)ptr;
		while ((*c++ = getc(file)) != EOF);
		printf("%s %d\n", ptr->name, ptr->age);
		free(ptr);
	} else return 1;
	return 0;
}

/* // For Linux\Windows OS // */

/* Start MAIN condition */
/* Operating System: Linux\Windows */
#if defined(__linux__) || defined(__WIN32__)

#include <stdio.h>

#define bool _Bool
enum{false, true};

bool create(const char *file) {
	FILE *ptrFile;
	if ((ptrFile = fopen(file, "w")) != NULL) {
		fclose(ptrFile);
		return true;
	} else {
		return false;
	}
}

bool input(const char *file, const char *text) {
	FILE *ptrFile;
	if ((ptrFile = fopen(file, "a")) != NULL) {
		fputs(text, ptrFile);
		fclose(ptrFile);
		return true;
	} else {
		return false;
	}
}

bool output(const char *file) {
	FILE *ptrFile;
	register char symbol;
	if ((ptrFile = fopen(file, "r")) != NULL) {
		while ((symbol = fgetc(ptrFile)) != EOF)
			printf("%c", symbol);
		fclose(ptrFile);
		return true;
	} else {
		return false;
	}
}

char *pwd(char *buffer) {
	FILE *result = popen("pwd", "r");
    if (result != NULL)
    	while(fgets(buffer, 128, result) != NULL);
    else return false;
    return buffer;
    pclose(result);
}

#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#define ENDSTRING 1
#define ENDCHAR '\0'
#define SPACESTRING 1
#define SPACECHAR ' '

#define RUNSIZE 2
void run(const char *path) {
	const char runSymbols[RUNSIZE] = "./";

	const unsigned char lengthString = strlen(path);

	char fullPath[RUNSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < RUNSIZE; fullPath[index++] = runSymbols[index]);
	for (unsigned char index = RUNSIZE, twindex = 0; index <= RUNSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[RUNSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define ECHOSIZE 5
void echo(const char *path) {
	const char echoSymbols[ECHOSIZE] = "echo ";

	const unsigned char lengthString = strlen(path);

	char fullPath[ECHOSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < ECHOSIZE; fullPath[index++] = echoSymbols[index]);
	for (unsigned char index = ECHOSIZE, twindex = 0; index <= ECHOSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[ECHOSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define DIRSIZE 4
void dir(const char *path) {
	const char lsSymbols[DIRSIZE] = "dir ";

	const unsigned char lengthString = strlen(path);
	
	char fullPath[DIRSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < DIRSIZE; fullPath[index++] = lsSymbols[index]);
	for (unsigned char index = DIRSIZE, twindex = 0; index <= DIRSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[DIRSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define MKDIRSIZE 6
void mkdir(const char *path) {
	const char mkdirSymbols[MKDIRSIZE] = "mkdir ";

	const unsigned char lengthString = strlen(path);

	char fullPath[MKDIRSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < MKDIRSIZE; fullPath[index++] = mkdirSymbols[index]);
	for (unsigned char index = MKDIRSIZE, twindex = 0; index <= MKDIRSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[MKDIRSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#endif 
/* End MAIN condition */


/* // Operation System: Linux // */
/* Start condition */
#if defined(__linux__)

void uname(void) { system("uname -a"); }

#define TOUCHSIZE 6
void touch(const char *path) {
	const char touchSymbols[TOUCHSIZE] = "touch ";

	const unsigned char lengthString = strlen(path);

	char fullPath[TOUCHSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < TOUCHSIZE; fullPath[index++] = touchSymbols[index]);
	for (unsigned char index = TOUCHSIZE, twindex = 0; index <= TOUCHSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[TOUCHSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define CATSIZE 4
void cat(const char *path) {
	const char catSymbols[CATSIZE] = "cat ";

	const unsigned char lengthString = strlen(path);

	char fullPath[CATSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < CATSIZE; fullPath[index++] = catSymbols[index]);
	for (unsigned char index = CATSIZE, twindex = 0; index <= CATSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[CATSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define FILESIZE 5
void file(const char *path) {
	const char fileSymbols[FILESIZE] = "file ";

	const unsigned char lengthString = strlen(path);

	char fullPath[FILESIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < FILESIZE; fullPath[index++] = fileSymbols[index]);
	for (unsigned char index = FILESIZE, twindex = 0; index <= FILESIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[FILESIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define MVSIZE 3
void mv(const char *pathOne, const char *pathTwo) {
	const char mvSymbols[MVSIZE] = "mv ";

	const unsigned char lengthString_One = strlen(pathOne);
	const unsigned char lengthString_Two = strlen(pathTwo);

	char fullPath[MVSIZE + lengthString_One + SPACESTRING + lengthString_Two + ENDSTRING];

	for (unsigned char index = 0; index < MVSIZE; fullPath[index++] = mvSymbols[index]);
	for (unsigned char index = MVSIZE, twindex = 0; index < MVSIZE+lengthString_One; fullPath[index++] = pathOne[twindex++]);

	fullPath[MVSIZE + lengthString_One] = SPACECHAR;

	for (unsigned char index = MVSIZE+lengthString_One+1, twindex = 0; index <= MVSIZE+lengthString_One+lengthString_Two; fullPath[index++] = pathTwo[twindex++]);

	fullPath[MVSIZE + lengthString_One + SPACESTRING + lengthString_Two] = ENDCHAR;

	system(fullPath);
}

#define CPSIZE 6
void cp(const char *pathOne, const char *pathTwo) {
	const char cpSymbols[CPSIZE] = "cp -r ";

	const unsigned char lengthString_One = strlen(pathOne);
	const unsigned char lengthString_Two = strlen(pathTwo);

	char fullPath[CPSIZE + lengthString_One + SPACESTRING + lengthString_Two + ENDSTRING];

	for (unsigned char index = 0; index < CPSIZE; fullPath[index++] = cpSymbols[index]);
	for (unsigned char index = CPSIZE, twindex = 0; index <= CPSIZE+lengthString_One; fullPath[index++] = pathOne[twindex++]);

	fullPath[CPSIZE + lengthString_One] = SPACECHAR;

	for (unsigned char index = CPSIZE+lengthString_One+1, twindex = 0; index <= CPSIZE+lengthString_One+lengthString_Two; fullPath[index++] = pathTwo[twindex++]);
	
	fullPath[CPSIZE + lengthString_One + SPACESTRING + lengthString_Two] = ENDCHAR;

	system(fullPath);
}

#define RMSIZE 7
void rm(const char *path) {
	const char rmSymbols[RMSIZE] = "rm -rf ";

	const unsigned char lengthString = strlen(path);

	char fullPath[RMSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < RMSIZE; fullPath[index++] = rmSymbols[index]);
	for (unsigned char index = RMSIZE, twindex = 0; index <= RMSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[RMSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}


/* // Operation System: Windows // */
/* Continue condition */
#elif defined(__WIN32__)

void clear(void) { system("cls"); }

#define COPYSIZE 8
void cp(const char *pathOne, const char *pathTwo) {
	const char cpSymbols[COPYSIZE] = "copy -r ";

	const unsigned char lengthString_One = strlen(pathOne);
	const unsigned char lengthString_Two = strlen(pathTwo);

	char fullPath[COPYSIZE + lengthString_One + SPACESTRING + lengthString_Two + ENDSTRING];

	for (unsigned char index = 0; index < COPYSIZE; fullPath[index++] = cpSymbols[index]);
	for (unsigned char index = COPYSIZE, twindex = 0; index <= COPYSIZE+lengthString_One; fullPath[index++] = pathOne[twindex++]);

	fullPath[COPYSIZE + lengthString_One] = SPACECHAR;

	for (unsigned char index = COPYSIZE+lengthString_One+1, twindex = 0; index <= COPYSIZE+lengthString_One+lengthString_Two; fullPath[index++] = pathTwo[twindex++]);
	
	fullPath[COPYSIZE + lengthString_One + SPACESTRING + lengthString_Two] = ENDCHAR;

	system(fullPath);
}

#define MOVESIZE 5
void mv(const char *pathOne, const char *pathTwo) {
	const char mvSymbols[MOVESIZE] = "move ";

	const unsigned char lengthString_One = strlen(pathOne);
	const unsigned char lengthString_Two = strlen(pathTwo);

	char fullPath[MOVESIZE + lengthString_One + SPACESTRING + lengthString_Two + ENDSTRING];

	for (unsigned char index = 0; index < MOVESIZE; fullPath[index++] = mvSymbols[index]);
	for (unsigned char index = MOVESIZE, twindex = 0; index <= MOVESIZE+lengthString_One; fullPath[index++] = pathOne[twindex++]);

	fullPath[MOVESIZE + lengthString_One] = SPACECHAR;

	for (unsigned char index = MOVESIZE+lengthString_One+1, twindex = 0; index <= MOVESIZE+lengthString_One+lengthString_Two; fullPath[index++] = pathTwo[twindex++]);
	
	fullPath[MOVESIZE + lengthString_One + SPACESTRING + lengthString_Two] = ENDCHAR;

	system(fullPath);
}

#define RDSIZE 8
void rd(const char *path) {
	const char rdSymbols[RDSIZE] = "rd /s/q ";

	const unsigned char lengthString = strlen(path);

	char fullPath[RDSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < RDSIZE; fullPath[index++] = rdSymbols[index]);
	for (unsigned char index = RDSIZE, twindex = 0; index <= RDSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[RDSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

/* // Operation System: Unknown // */
#else
#warning "Warning: Unknown OS."

/* End condition */
#endif 

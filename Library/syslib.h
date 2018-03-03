#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define bool _Bool
#define true 1
#define false 0

#define ENDSTRING 1
#define ENDCHAR '\0'
#define SPACESTRING 1
#define SPACECHAR ' '

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

/* // For Linux\Windows OS // */

/* Start condition */
#if defined(__linux__) || defined(__WIN32__)

#define RUN "./"
#define RUNSIZE 2
void run(const char *path) {
	const char runSymbols[RUNSIZE] = RUN;

	const unsigned char lengthString = strlen(path);

	char fullPath[RUNSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < RUNSIZE; fullPath[index++] = runSymbols[index]);
	for (unsigned char index = RUNSIZE, twindex = 0; index <= RUNSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[RUNSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define ECHO "echo "
#define ECHOSIZE 5
void echo(const char *path) {
	const char echoSymbols[ECHOSIZE] = ECHO;

	const unsigned char lengthString = strlen(path);

	char fullPath[ECHOSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < ECHOSIZE; fullPath[index++] = echoSymbols[index]);
	for (unsigned char index = ECHOSIZE, twindex = 0; index <= ECHOSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[ECHOSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define DIR "dir "
#define DIRSIZE 4
void dir(const char *path) {
	const char lsSymbols[DIRSIZE] = DIR;

	const unsigned char lengthString = strlen(path);
	
	char fullPath[DIRSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < DIRSIZE; fullPath[index++] = lsSymbols[index]);
	for (unsigned char index = DIRSIZE, twindex = 0; index <= DIRSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[DIRSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define MKDIR "mkdir "
#define MKDIRSIZE 6
void mkdir(const char *path) {
	const char mkdirSymbols[MKDIRSIZE] = MKDIR;

	const unsigned char lengthString = strlen(path);

	char fullPath[MKDIRSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < MKDIRSIZE; fullPath[index++] = mkdirSymbols[index]);
	for (unsigned char index = MKDIRSIZE, twindex = 0; index <= MKDIRSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[MKDIRSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#endif 
/* End condition */


/* // Operation System: Linux // */

/* Start condition */
#if defined(__linux__)

#include <unistd.h>

void clear(void) { system("clear"); }

void pwd(void) { system("pwd"); }

void uname(void) { system("uname -a"); }

#define TOUCH "touch "
#define TOUCHSIZE 6
void touch(const char *path) {
	const char touchSymbols[TOUCHSIZE] = TOUCH;

	const unsigned char lengthString = strlen(path);

	char fullPath[TOUCHSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < TOUCHSIZE; fullPath[index++] = touchSymbols[index]);
	for (unsigned char index = TOUCHSIZE, twindex = 0; index <= TOUCHSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[TOUCHSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define CAT "cat "
#define CATSIZE 4
void cat(const char *path) {
	const char catSymbols[CATSIZE] = CAT;

	const unsigned char lengthString = strlen(path);

	char fullPath[CATSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < CATSIZE; fullPath[index++] = catSymbols[index]);
	for (unsigned char index = CATSIZE, twindex = 0; index <= CATSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[CATSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define FILE "file "
#define FILESIZE 5
void file(const char *path) {
	const char fileSymbols[FILESIZE] = FILE;

	const unsigned char lengthString = strlen(path);

	char fullPath[FILESIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < FILESIZE; fullPath[index++] = fileSymbols[index]);
	for (unsigned char index = FILESIZE, twindex = 0; index <= FILESIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[FILESIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define MV "mv "
#define MVSIZE 3
void mv(const char *pathOne, const char *pathTwo) {
	const char mvSymbols[MVSIZE] = MV;

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

#define CP "cp "
#define CPSIZE 3
void cp(const char *pathOne, const char *pathTwo) {
	const char cpSymbols[CPSIZE] = CP;

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

#define RM "rm -rf "
#define RMSIZE 7
void rm(const char *path) {
	const char rmSymbols[RMSIZE] = RM;

	const unsigned char lengthString = strlen(path);

	char fullPath[RMSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < RMSIZE; fullPath[index++] = rmSymbols[index]);
	for (unsigned char index = RMSIZE, twindex = 0; index <= RMSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[RMSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define LS "ls -l "
#define LSZISE 6
void ls(const char *path) {
	const char lsSymbols[LSZISE] = LS;

	const unsigned char lengthString = strlen(path);
	
	char fullPath[LSZISE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < LSZISE; fullPath[index++] = lsSymbols[index]);
	for (unsigned char index = LSZISE, twindex = 0; index <= LSZISE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[LSZISE + lengthString] = ENDCHAR;

	system(fullPath);
}

/* // Operation System: Windows // */

/* Continue condition */
#elif defined(__WIN32__)

void cls(void) { system("cls"); }

#define ECHO "echo "
#define ECHOSIZE 5
void echo(const char *path) {
	const char echoSymbols[ECHOSIZE] = ECHO;

	const unsigned char lengthString = strlen(path);

	char fullPath[ECHOSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < ECHOSIZE; fullPath[index++] = echoSymbols[index]);
	for (unsigned char index = ECHOSIZE, twindex = 0; index <= ECHOSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[ECHOSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define COPY "copy -r "
#define COPYSIZE 8
void copy(const char *pathOne, char *pathTwo) {
	const char cpSymbols[COPYSIZE] = COPY;

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

#define MOVE "move "
#define MOVESIZE 5
void move(const char *pathOne, const char *pathTwo) {
	const char moveSymbols[MOVESIZE] = MOVE;

	const unsigned char lengthString_One = strlen(pathOne);
	const unsigned char lengthString_Two = strlen(pathTwo);

	char fullPath[MOVESIZE + lengthString_One + SPACESTRING + lengthString_Two + ENDSTRING];

	for (unsigned char index = 0; index < MOVESIZE; fullPath[index++] = moveSymbols[index]);
	for (unsigned char index = MOVESIZE, twindex = 0; index <= MOVESIZE+lengthString_One; fullPath[index++] = pathOne[twindex++]);

	fullPath[MOVESIZE + lengthString_One] = SPACECHAR;

	for (unsigned char index = MOVESIZE+lengthString_One+1, twindex = 0; index <= MOVESIZE+lengthString_One+lengthString_Two; fullPath[index++] = pathTwo[twindex++]);
	
	fullPath[MOVESIZE + lengthString_One + SPACESTRING + lengthString_Two] = ENDCHAR;

	system(fullPath);
}

#define REN "ren "
#define RENSIZE 4
void ren(const char *pathOne, const char *pathTwo) {
	const char renSymbols[RENSIZE] = REN;

	const unsigned char lengthString_One = strlen(pathOne);
	const unsigned char lengthString_Two = strlen(pathTwo);

	char fullPath[RENSIZE + lengthString_One + SPACESTRING + lengthString_Two + ENDSTRING];

	for (unsigned char index = 0; index < RENSIZE; fullPath[index++] = renSymbols[index]);
	for (unsigned char index = RENSIZE, twindex = 0; index <= RENSIZE+lengthString_One; fullPath[index++] = pathOne[twindex++]);

	fullPath[RENSIZE + lengthString_One] = SPACECHAR;

	for (unsigned char index = RENSIZE+lengthString_One+1, twindex = 0; index <= RENSIZE+lengthString_One+lengthString_Two; fullPath[index++] = pathTwo[twindex++]);
	
	fullPath[RENSIZE + lengthString_One + SPACESTRING + lengthString_Two] = ENDCHAR;

	system(fullPath);
}

#define DEL "del "
#define DELSIZE 4
void del(const char *path) {
	const char delSymbols[DELSIZE] = DEL;

	const unsigned char lengthString = strlen(path);

	char fullPath[DELSIZE + lengthString + ENDSTRING];

	for (unsigned char index = 0; index < DELSIZE; fullPath[index++] = delSymbols[index]);
	for (unsigned char index = DELSIZE, twindex = 0; index <= DELSIZE+lengthString; fullPath[index++] = path[twindex++]);

	fullPath[DELSIZE + lengthString] = ENDCHAR;

	system(fullPath);
}

#define RD "rd /s/q "
#define RDSIZE 8
void rd(const char *path) {
	const char rdSymbols[RDSIZE] = RD;

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

#endif 
/* End condition */

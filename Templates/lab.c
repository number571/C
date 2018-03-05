// https://www.onlinegdb.com/online_c_compiler
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define bool _Bool
#define true 1
#define false 0

void oneFunc(char text[]);
void twoFunc(char text[]);
void threeFunc(char text[]);
void fourFunc(char text[]);
void fiveFunc(char text[]);

int main(void) {
	char text[] = "Hello World! More than you know.";
	oneFunc(text);
	return 0;
}

// Инверсия слов в тексте
void fiveFunc(char text[]) {
	unsigned short lengthString = strlen(text) + 1;
	char beforeText[lengthString]; beforeText[0] = ' ';
	for (short index = 1, twindex = 0; index < lengthString; beforeText[index++] = text[twindex++]);
	char afterText[lengthString];
	char finalText[lengthString];
	unsigned short counter = 0, position = 0, thrindex = 0;
	for (short index = lengthString, twindex = 0; index >= 0; afterText[twindex++] = beforeText[index--]);
	for (unsigned short index = 0; index <= lengthString; index++) {
		if (afterText[index] == ' ' || index == lengthString) {
			for (short twindex = counter; twindex > position; ) {
				finalText[thrindex++] = afterText[twindex--];
			}
			position = counter++;
		} else {
			counter ++;
		}
	}
	for (unsigned short index = 0; index < lengthString; index++) {
	    printf("%c", finalText[index]);
	}
}

// Количество одинаковых слов в тексте
void fourFunc(char text[]) {
	unsigned short lengthString = strlen(text);
	char word[] = "than";
	unsigned short counter = 0;
	for (unsigned short index, twindex = 0; index < lengthString; index++) {
		if (text[index] == word[twindex++]) {
			if (twindex == strlen(word))
				counter ++;
		} else {
			twindex = 0;
		}
	}
	printf("%hu\n", counter);
}

// Дублирование указанного слова в тексте
void threeFunc(char text[]) {
	unsigned short choiceNumber = 1;
	unsigned short lengthWord = 0;
	unsigned short lengthString = strlen(text);
	for (unsigned short index, counter = 0; index < lengthString; index++) {
		if (text[index] == ' ' && text[index+1] != ' '){
			counter++;
			continue;
		} else {
			if (isalpha(text[index]) && choiceNumber == counter)
				lengthWord++;
		}
	}
	char word[lengthWord];
	char afterText[lengthString+lengthWord];
	unsigned short quanity = 0;
	for (unsigned short index, twindex, thrindex, counter = 0; index < lengthString; index++, quanity++) {
		if (text[index] == ' ' && text[index+1] != ' '){
			afterText[twindex++] = text[index];
			counter++;
			continue;
		} else {
			if (choiceNumber < counter) {
				break;
			}
			if (isalpha(text[index]) && choiceNumber == counter) {
				word[thrindex++] = text[index];
			}
			afterText[twindex++] = text[index];
		}
	}
	for (unsigned short index = 0; index < lengthWord; index++)
		afterText[quanity+index] = word[index];
	for (unsigned short index = quanity-1, twindex = 0; index < lengthString; index++, twindex++)
		afterText[quanity+lengthWord+twindex] = text[index];
	for (unsigned short index = 0; index < lengthString+lengthWord+1; index++) {
	    printf("%c",afterText[index]);
	}
}

// Удаление указанного слова
void twoFunc(char text[]) {
	unsigned short lengthString = strlen(text);
	unsigned short choiceNumber = 2;
	for (unsigned short index, counter = 0; index < lengthString; index++) {
		if (text[index] == ' '){
			counter++;
			continue;
		} else {
			if (isalpha(text[index]) && choiceNumber == counter)
				text[index] = ' ';
		}
	}
	printf("%s\n", text);
}

// Проверка символа в указанном слове
void oneFunc(char text[]) {
	bool switcher = false;
	unsigned short lengthString = strlen(text);
	unsigned short choiceNumber = 0;
	char choiceSymbol = 'e';
	for (unsigned short index, counter = 0; index < lengthString; index++) {
		if (text[index] == ' '){
			counter++;
			continue;
		} else {
			if (isalpha(text[index])) {
				if (choiceNumber == counter && choiceSymbol == text[index]) {
					switcher = true;
					break;
				}
			}	
		}
	}
	printf("%s\n", switcher?"Yes":"No");
}

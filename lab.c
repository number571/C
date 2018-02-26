#include <stdio.h>
#include <string.h>
#include <ctype.h>

void oneFunc(char text[]);
void twoFunc(char text[]);
void threeFunc(char text[]);
void fourFunc(char text[]);
void fiveFunc(char text[]);


int main(void) {
	char text[] = "More than you know.";
	fiveFunc(text);
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
	puts(finalText);
}

// Количество одинаковых слов в тексте
void fourFunc(char text[]) {
	char word[] = "than";
	unsigned short counter = 0;
	for (unsigned short index, twindex = 0; index < strlen(text); index++) {
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
	for (unsigned short index, twindex, thrindex, counter = 0; index < strlen(text); index++, quanity++) {
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
	for (unsigned short index = quanity-1, twindex = 0; index < strlen(text); index++, twindex++)
		afterText[quanity+lengthWord+twindex] = text[index];
	printf("%s\n", afterText);
}

// Удаление указанного слова
void twoFunc(char text[]) {
	unsigned short choiceNumber = 2;
	for (unsigned short index, counter = 0; index < strlen(text); index++) {
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
	unsigned short choiceNumber = 1;
	char choiceSymbol = 'a';
	for (unsigned short index, counter = 0; index < strlen(text); index++) {
		if (text[index] == ' ' && text[index+1] != ' '){
			counter++;
			continue;
		} else {
			if (isalpha(text[index]))
				if (choiceNumber == counter && choiceSymbol == text[index]) {
					printf("Yes\n");
					break;
				}
		}
	}
}
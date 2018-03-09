#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define bool _Bool
enum {false, true};

int main(void) {
	bool value = true;
	srand(time(NULL)); 
	printf("--------------------------\n");
	while(value) {
		short choice, opponent = 1 + rand() % 3;
		printf("1 - Rock\n2 - Scissors\n3 - Paper\n> ");
		scanf("%hd", &choice); 
		printf("--------------------------\n");
		printf("Opponent choice: %s\n",(opponent == 1)?"Rock":(opponent == 2)?"Scissors":"Paper");
		if (choice == opponent) {
			printf("== Draw! ==\n");
		} else if (choice == 1) {
			if (opponent == 2)
				printf("++ You win! ++\n");
			else
				printf("-- Opponent win! --\n");
		} else if (choice == 2) {
			if (opponent == 3)
				printf("++ You win! ++\n");
			else
				printf("-- Opponent win! --\n");
		} else if (choice == 3) {
			if (opponent == 1)
				printf("++ You win! ++\n");
			else
				printf("-- Opponent win! --\n");
		} else {
			printf("Number is not defined!\n");
			value = false;
		} printf("--------------------------\n");
	}
	return 0;
}
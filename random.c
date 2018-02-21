#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(void) {
	srand(time(NULL));
	for (short i = 0; i < 10; i ++) {
		printf("%d\n", rand()%100);
	}
	return 0;
}
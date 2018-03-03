#include <stdio.h>
long factorial(int x);

int main(void) {
	int number; scanf("%d", &number);
	printf("%ld\n",factorial(number));
	return 0;
}

long factorial(int x) {
	if (x < 1)
		return 1;
	else
		return x * factorial(x-1);
}

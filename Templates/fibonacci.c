#include <stdio.h>
long fibonacci(int x);

int main(void) {
	int number; scanf("%d",&number);
	printf("%ld\n", fibonacci(number));
	return 0;
}

long fibonacci(int x) {
	if (x < 1)
		return 1;
	else
		return fibonacci(x - 1) + fibonacci(x - 2);
}

#include <stdio.h>
void moveTo(int *x, int *y);

int main(void) {
	int x, y;
	scanf("%d %d", &x, &y);
	moveTo(&x, &y);
	printf("%d %d\n", x,y);
}

void moveTo(int *x, int *y) {
	*x = *x ^ *y;
	*y = *x ^ *y;
	*x = *x ^ *y;
}

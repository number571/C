#include <stdio.h>
double power(float x, short y);

int main(void) {
	float number; scanf("%f", &number);
	short powNumber; scanf("%hd", &powNumber);
	printf("%.2lf\n",power(number, powNumber));
	return 0;
}

double power(float x, short y) {
	if (y == 0)
		return 1;
	else
		return x * power(x,y-1);
}
#include <stdio.h>
#include <stdbool.h>
int main(void) {
	double x,y; char symbol;
	while(true){
		printf("Calculator: ");
		scanf("%lf %c %lf",&x, &symbol, &y);
		switch(symbol){
			case '+': printf("Result: %.2lf\n", x+y); break;
			case '-': printf("Result: %.2lf\n", x-y); break;
			case '*': printf("Result: %.2lf\n", x*y); break;
			case '/': printf("Result: %.2lf\n", x/y); break;
			default: printf("Error! Operator not found!\n");
		}
	}
	return 0;
}
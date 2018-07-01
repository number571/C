#include <stdio.h>

union code {
	int number;
	struct {
		unsigned _0: 1;
		unsigned _1: 1;
		unsigned _2: 1;
		unsigned _3: 1;
		unsigned _4: 1;
		unsigned _5: 1;
		unsigned _6: 1;
		unsigned _7: 1;
	} byte;
};

int main(void) {
	union code check;
	check.number = 5;

	printf("%d %d %d %d %d %d %d %d\n", 
		check.byte._7, check.byte._6, check.byte._5, check.byte._4,
		check.byte._3, check.byte._2, check.byte._1, check.byte._0);

    return 0;
}

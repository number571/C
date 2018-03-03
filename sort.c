#include <stdio.h>
#define SIZE 10

short sort(short *array);
short reverse(short *array);
void output(short *array);

int main(void) {
	short arr[SIZE] = {4,3,8,5,1,0,9,6,7,2};
	sort(arr); reverse(arr); output(arr);
	return 0;
}

short sort(short *array) {
	short temp;
	for (unsigned short index = 1; index < SIZE; index ++) {
		for (unsigned short twindex = 1; twindex < SIZE; twindex ++) {
			if (array[twindex] < array[twindex - 1]) {
				temp = array[twindex];
				array[twindex] = array[twindex - 1];
				array[twindex - 1] = temp;
			}
		}
	}
}

short reverse(short *array) {
	short temp;
	for (unsigned short index = 0; index < SIZE/2; index ++) {
		temp = array[index];
		array[index] = array[SIZE - index - 1];
		array[SIZE - index - 1] = temp;
	}
}
void output(short *array) {
	for (unsigned short index = 0; index < SIZE; index ++) {
		printf("%hd\n", array[index]);
	}
}
#include <stdio.h>
#define SIZE 10
_Bool check(const char arr[], const char choice, char index) {
	if (index < 0)
		return 0;
	else if (arr[index] == choice)
		return 1;
	else
		return check(arr, choice, --index);
}
int main(void) {
	const char array[SIZE] = {5,4,3,8,0,9,7,1,2,6};
	printf("%s\n", check(array, 6, SIZE-1)?"Yes":"No");
	return 0;
}
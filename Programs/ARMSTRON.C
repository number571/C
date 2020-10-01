#include<stdio.h>
#include<conio.h>

void main()
{
	int num,temp,count=0,;

	printf("Enter a number = ");
	scanf("%d",&num);

	temp = num;

	while(num != 0)
	{
		num = num % 10;
		count++;
	}

	while(count>0)
	{

	}
}
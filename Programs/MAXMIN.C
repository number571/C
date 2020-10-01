#include<stdio.h>
#include<conio.h>

void main()
{
	int i,num,max=1,min;

	clrscr();


	for(i=1;i<=4;i++)
	{
		printf("Enter a number = ");
		scanf("%d", &num);


		if(num > max)
		{
			max = num;
		}
		if(num < min)
		{
			min = num;
		}
	}
	printf("The largest number is = %d\nThe smallest number is = %d\n",max,min);

	getch();


}
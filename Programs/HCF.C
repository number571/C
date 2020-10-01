#include<stdio.h>
#include<conio.h>
main()
{
	int a,b,i;

	clrscr();

	printf("Enter 2 numbers = ");
	scanf("%d %d",&a,&b);

	for(i=a<b?a:b;i>=1;i--)
	{
		if(a%i==0 && b%i==0)
			break;
	}
	printf("HCF = %d",i);

	getch();

}
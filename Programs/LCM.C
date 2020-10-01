main()
{
	int a,b,i;

	clrscr();

	printf("Enter 2 numbers = ");
	scanf("%d %d",&a,&b);

	for(i=2;i<=a*b;i++)
	{
		if(i%a==0 && i%b==0)
		break;
	}
	printf("%d",i);

	getch();


}
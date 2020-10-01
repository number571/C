#define size1 5
#define size2 6
#include<stdio.h>
#include<conio.h>
void input(int *ptr,int size)
{
	int i;
	for(i=0;i<size;i++)
	{
		printf("Enter %d number =",i+1);
		scanf("%d",(ptr+i));
	}
}
void output(int *ptr,int size)
{
	int i;

	for(i=0;i<size;i++)
	{
		printf("%d\t",*(ptr+i));
	}
	printf("\n");

}
void sort(int *ptr,int size)
{
	int i,j;

	for(i=0;i<size-1;i++)
	{
		for(j=i+1;j<size;j++)
		{
			if(*(ptr+i)>*(ptr+j))
			{
				int temp;
				temp     = *(ptr+i);
				*(ptr+i) = *(ptr+j);
				*(ptr+j) = temp;
			}
		}
	}

}
void merge(int *ptr1,int *ptr2,int *ptr3)
{
	int i=0,j=0,k=0;

	while(i<size1 && j<size2)
	{
		if(*(ptr1+i)<*(ptr2+j))
		{
			*(ptr3+k) = *(ptr1+i);
			i++; k++;
		}
		else
		{
			*(ptr3+k) = *(ptr2+j);
			j++;  k++;
		}

	}
	while(i<size1)
	{
		*(ptr3+k) = *(ptr1+i);
		k++;i++;
	}
	while(j<size2)
	{
		*(ptr3+k) = *(ptr2+j);
		k++ ; j++;
	}
}



int main()
{
	int num1[size1],num2[size2],num3[size1+size2];


	input(num1,size1);
	input(num2,size2);

	output(num1,size1);
	output(num2,size2);

	sort(num1,size1);
	sort(num2,size2);

	printf("After sorting\n");

	output(num1,size1);
	output(num2,size2);

	merge(num1,num2,num3);

	printf("Merge output\n");
	output(num3,size1+size2);

	getch();

}
